"""Download the LIBERO HDF5 demonstrations selected by a data config."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
import yaml


REPO_ID = "yifengzhu-hf/LIBERO-datasets"
def _requested_files(data_config_path: Path) -> list[str]:
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    data_config = yaml.safe_load(data_config_path.read_text())
    requested_files = []
    for split_name in ("seen", "goal"):
        split = data_config[split_name]
        suite = split["suite"]
        for task_id in split["task_ids"]:
            task_name = libero_task_map[suite][task_id]
            requested_files.append(f"{suite}/{task_name}_demo.hdf5")
    return requested_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/libero_hdf5"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/libero.yaml"))
    args = parser.parse_args()

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.root,
        allow_patterns=_requested_files(args.data_config),
        max_workers=4,
    )


if __name__ == "__main__":
    main()
