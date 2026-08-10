"""On-the-fly PyTorch dataset for canonical LIBERO HDF5 demonstrations."""

from bisect import bisect_right
from dataclasses import dataclass
from functools import cache
import importlib.util
import json
from pathlib import Path
import re

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


DEMO_NAME = re.compile(r"demo_(\d+)")
CAMERA1_KEY = "observation.images.camera1"
CAMERA2_KEY = "observation.images.camera2"
TASK_MAP_PATH = (
    Path(__file__).resolve().parents[1]
    / "third_party/LIBERO/libero/libero/benchmark/libero_suite_task_map.py"
)


@dataclass(frozen=True)
class Episode:
    source_path: Path
    task_id: int
    instruction: str
    demo_id: int
    length: int


def _sorted_demo_ids(hdf5: h5py.File) -> list[int]:
    demo_ids = []
    for key in hdf5["data"]:
        match = DEMO_NAME.fullmatch(key)
        if match is None:
            raise ValueError(f"Unexpected episode key: {key}")
        demo_ids.append(int(match.group(1)))
    demo_ids.sort()
    if demo_ids != list(range(50)):
        raise ValueError(f"Expected demo IDs 0--49, found {demo_ids}")
    return demo_ids


@cache
def _task_map() -> dict[str, list[str]]:
    spec = importlib.util.spec_from_file_location("libero_suite_task_map", TASK_MAP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.libero_task_map


def _validate_episode(episode: h5py.Group, source_path: Path, demo_id: int) -> int:
    num_frames = int(episode.attrs["num_samples"])
    expected_shapes = {
        "actions": (num_frames, 7),
        "obs/ee_pos": (num_frames, 3),
        "obs/ee_ori": (num_frames, 3),
        "obs/gripper_states": (num_frames, 2),
        "obs/agentview_rgb": (num_frames, 128, 128, 3),
        "obs/eye_in_hand_rgb": (num_frames, 128, 128, 3),
    }
    for key, expected_shape in expected_shapes.items():
        if episode[key].shape != expected_shape:
            raise ValueError(
                f"Invalid shape in {source_path}: demo_{demo_id}/{key} "
                f"has {episode[key].shape}, expected {expected_shape}"
            )
    return num_frames


class LiberoHDF5Dataset(Dataset):
    """Loads LIBERO frames directly from HDF5 without duplicating image data on disk."""

    def __init__(
        self,
        root: str | Path,
        suite: str,
        task_ids: list[int],
        demos_per_task: int,
        action_chunk_size: int = 50,
    ) -> None:
        self.action_chunk_size = action_chunk_size
        self.episodes = []
        self._files: dict[Path, h5py.File] = {}

        for task_id in task_ids:
            task_name = _task_map()[suite][task_id]
            source_path = Path(root) / suite / f"{task_name}_demo.hdf5"
            with h5py.File(source_path, "r") as hdf5:
                instruction = json.loads(hdf5["data"].attrs["problem_info"])["language_instruction"]
                demo_ids = _sorted_demo_ids(hdf5)[:demos_per_task]
                for demo_id in demo_ids:
                    episode = hdf5["data"][f"demo_{demo_id}"]
                    self.episodes.append(
                        Episode(
                            source_path=source_path,
                            task_id=task_id,
                            instruction=instruction,
                            demo_id=demo_id,
                            length=_validate_episode(episode, source_path, demo_id),
                        )
                    )

        self._episode_ends = np.cumsum([episode.length for episode in self.episodes]).tolist()

    def __len__(self) -> int:
        return self._episode_ends[-1]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        episode_index = bisect_right(self._episode_ends, index)
        previous_end = 0 if episode_index == 0 else self._episode_ends[episode_index - 1]
        frame_index = index - previous_end
        episode_info = self.episodes[episode_index]
        episode = self._open_file(episode_info.source_path)["data"][f"demo_{episode_info.demo_id}"]
        observation = episode["obs"]

        state = np.concatenate(
            [
                observation["ee_pos"][frame_index],
                observation["ee_ori"][frame_index],
                observation["gripper_states"][frame_index],
            ]
        ).astype(np.float32)
        return {
            CAMERA1_KEY: self._image_tensor(observation["agentview_rgb"][frame_index]),
            CAMERA2_KEY: self._image_tensor(observation["eye_in_hand_rgb"][frame_index]),
            "observation.state": torch.from_numpy(state),
            **self._action_chunk(episode["actions"], frame_index),
            "task": episode_info.instruction,
        }

    def seen_dataset_stats(self) -> dict[str, dict[str, torch.Tensor]]:
        """Compute normalization statistics from the selected seen demonstrations."""
        values = {"observation.state": [], "action": []}
        episodes_by_path: dict[Path, list[Episode]] = {}
        for episode_info in self.episodes:
            episodes_by_path.setdefault(episode_info.source_path, []).append(episode_info)

        for source_path, source_episodes in episodes_by_path.items():
            with h5py.File(source_path, "r") as hdf5:
                for episode_info in source_episodes:
                    episode = hdf5["data"][f"demo_{episode_info.demo_id}"]
                    values["observation.state"].append(
                        np.concatenate(
                            [
                                episode["obs/ee_pos"][:],
                                episode["obs/ee_ori"][:],
                                episode["obs/gripper_states"][:],
                            ],
                            axis=1,
                        )
                    )
                    values["action"].append(episode["actions"][:])

        stats = {}
        for key, parts in values.items():
            combined = np.concatenate(parts)
            stats[key] = {
                "mean": torch.from_numpy(combined.mean(axis=0).astype(np.float32)),
                "std": torch.from_numpy(combined.std(axis=0).astype(np.float32)),
            }
        return stats

    def source_manifest(self) -> list[dict]:
        return [
            {
                "task_id": episode.task_id,
                "source_file": str(episode.source_path),
                "source_demo_id": episode.demo_id,
                "instruction": episode.instruction,
            }
            for episode in self.episodes
        ]

    def close(self) -> None:
        for hdf5 in self._files.values():
            hdf5.close()
        self._files.clear()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_files"] = {}
        return state

    def _open_file(self, path: Path) -> h5py.File:
        if path not in self._files:
            self._files[path] = h5py.File(path, "r")
        return self._files[path]

    def _action_chunk(self, actions: h5py.Dataset, frame_index: int) -> dict[str, torch.Tensor]:
        available_actions = np.asarray(actions[frame_index : frame_index + self.action_chunk_size], dtype=np.float32)
        valid_length = len(available_actions)
        action_chunk = np.zeros((self.action_chunk_size, 7), dtype=np.float32)
        action_chunk[:valid_length] = available_actions
        is_pad = np.arange(self.action_chunk_size) >= valid_length
        return {
            "action": torch.from_numpy(action_chunk),
            "actions_is_pad": torch.from_numpy(is_pad),
        }

    @staticmethod
    def _image_tensor(image: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)
        # LeRobot applies the same 180° correction to observations emitted by LIBERO.
        return torch.flip(tensor, dims=(1, 2)).float().div_(255)
