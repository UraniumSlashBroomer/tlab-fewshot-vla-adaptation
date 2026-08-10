"""Download the canonical LIBERO HDF5 demonstrations used in this project."""

from pathlib import Path

from huggingface_hub import snapshot_download
import yaml


REPO_ID = "yifengzhu-hf/LIBERO-datasets"
DATA_ROOT = Path("data/libero_hdf5")
DATA_CONFIG = Path("configs/data/libero.yaml")


def _requested_files() -> list[str]:
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    data_config = yaml.safe_load(DATA_CONFIG.read_text())
    requested_files = []
    for split_name in ("seen", "goal"):
        split = data_config[split_name]
        suite = split["suite"]
        for task_id in split["task_ids"]:
            task_name = libero_task_map[suite][task_id]
            requested_files.append(f"{suite}/{task_name}_demo.hdf5")
    return requested_files


def main() -> None:
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=DATA_ROOT,
        allow_patterns=_requested_files(),
        max_workers=4,
    )


if __name__ == "__main__":
    main()
