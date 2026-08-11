"""Create and reset one LIBERO environment to validate local rendering."""

from gymnasium.vector import SyncVectorEnv
from lerobot.envs.libero import create_libero_envs


def main() -> None:
    envs = create_libero_envs(
        task="libero_goal",
        n_envs=1,
        env_cls=SyncVectorEnv,
        gym_kwargs={
            "task_ids": [0],
            "obs_type": "pixels_agent_pos",
            "observation_height": 128,
            "observation_width": 128,
        },
    )
    env = envs["libero_goal"][0]
    observation, _ = env.reset()
    print("reset passed; observation keys:", list(observation))
    env.close()
    print("LIBERO WGL smoke passed")


if __name__ == "__main__":
    main()
