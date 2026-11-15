"""Change action-expert depth while preserving SmolVLA's layer-specific VLM context."""

import random
from types import MethodType

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel


def get_depth_schedule(policy):
    base_policy = policy.get_base_model() if hasattr(policy, "peft_config") else policy
    return getattr(base_policy.model.vlm_with_expert, "depth_schedule", None)


class DepthSchedule:
    def __init__(self, mode: str, num_layers: int, seed: int):
        if mode not in {"skipping", "repetition"}:
            raise ValueError(f"Unknown depth method: {mode}")
        self.mode = mode
        self.num_layers = num_layers
        self.rng = random.Random(seed)
        self.selected_layers = ()
        self.episode_records = []

    def sample_training_batch(self):
        n = self.rng.choice((0, 1, 2))
        self.selected_layers = tuple(sorted(self.rng.sample(range(self.num_layers), n)))

    def begin_episode(self):
        episode_id = len(self.episode_records)
        # Independent of training seed, method, budget, and policy-noise RNG.
        rng = random.Random(f"{self.eval_seed}:{self.task_id}:{episode_id}:{self.eval_n}")
        self.selected_layers = tuple(sorted(rng.sample(range(self.num_layers), self.eval_n)))
        self.episode_records.append(
            {"episode_id": episode_id, "selected_layers": list(self.selected_layers)}
        )


def _depth_forward(
    self,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    fill_kv_cache=None,
):
    schedule = self.depth_schedule
    prefix, hidden = inputs_embeds
    if hidden is None or not schedule.selected_layers:
        return SmolVLMWithExpertModel.forward(
            self, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, fill_kv_cache
        )

    prefix_out = None
    if prefix is not None:
        prefix_len = prefix.shape[1]
        # Prefix tokens cannot attend to actions. Compute them once, retaining the
        # autograd graph through the cache (in particular, through state_proj).
        (prefix_out, _), past_key_values = SmolVLMWithExpertModel.forward(
            self,
            attention_mask=attention_mask[:, :prefix_len, :prefix_len],
            position_ids=position_ids[:, :prefix_len],
            inputs_embeds=[prefix, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        attention_mask = attention_mask[:, prefix_len:, :]
        position_ids = position_ids[:, prefix_len:]

    model_layers = self.get_model_layers([self.get_vlm_model().text_model, self.lm_expert])
    for layer_idx, layer in enumerate(self.lm_expert.layers):
        selected = layer_idx in schedule.selected_layers
        if selected and schedule.mode == "skipping":
            continue
        repeats = 2 if selected and schedule.mode == "repetition" else 1
        is_self_attention = layer_idx % self.self_attn_every_n_layers == 0
        attention = self.forward_attn_layer if is_self_attention else self.forward_cross_attn_layer
        for _ in range(repeats):
            # Use the original layer index and read-only prefix KV at BOTH calls.
            # Suffix KV is never written back, so repetition cannot grow the cache.
            outputs, _ = attention(
                model_layers,
                [None, hidden],
                layer_idx,
                position_ids,
                attention_mask,
                hidden.shape[0],
                self.config.text_config.head_dim,
                use_cache=True,
                fill_kv_cache=False,
                past_key_values=past_key_values,
            )
            # With prefix=None both upstream attention functions return only
            # the expert output, including the cached cross-attention path.
            att_output = outputs[0]
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
            hidden = layer.self_attn.o_proj(att_output) + hidden
            hidden = layer.mlp(layer.post_attention_layernorm(hidden)) + hidden

    return [prefix_out, self.lm_expert.norm(hidden)], past_key_values


def configure_depth(policy, mode: str, seed: int = 0) -> DepthSchedule:
    model = policy.model.vlm_with_expert
    if model.num_expert_layers != model.num_vlm_layers:
        raise ValueError("Depth experiments require one expert block per VLM layer.")
    if "cross" not in model.attention_mode or model.self_attn_every_n_layers != 2:
        raise ValueError("Depth experiments require alternating SA/CA SmolVLA blocks.")
    if not policy.config.use_cache or policy.config.compile_model:
        raise ValueError("Depth experiments require use_cache=true and compile_model=false.")
    schedule = DepthSchedule(mode, model.num_expert_layers, seed)
    model.depth_schedule = schedule
    # Keep the original modules and state_dict keys; checkpoints remain full
    # SmolVLA checkpoints, with a small sidecar describing the execution method.
    model.forward = MethodType(_depth_forward, model)
    return schedule


def _reset_with_depth(self):
    SmolVLAPolicy.reset(self)
    self.model.vlm_with_expert.depth_schedule.begin_episode()


def configure_depth_evaluation(policy, n: int, seed: int, task_id: int):
    if n not in (0, 1, 2):
        raise ValueError("Evaluation depth n must be 0, 1, or 2.")
    schedule = policy.model.vlm_with_expert.depth_schedule
    schedule.eval_n = n
    schedule.eval_seed = seed
    schedule.task_id = task_id
    schedule.episode_records = []
    schedule.selected_layers = ()
    # LeRobot calls policy.reset once per rollout; eval.py enforces num_envs=1.
    policy.reset = MethodType(_reset_with_depth, policy)
