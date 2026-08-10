"""Create LIBERO's non-interactive local path configuration."""

import argparse
from pathlib import Path

import yaml


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, default=Path("third_party/LIBERO"))
parser.add_argument("--dataset", type=Path, default=Path("data/libero_hdf5"))
parser.add_argument("--config", type=Path, default=Path(".libero/config.yaml"))
args = parser.parse_args()

benchmark_root = (args.source / "libero" / "libero").resolve()
args.config.parent.mkdir(parents=True, exist_ok=True)
args.config.write_text(
    yaml.safe_dump(
        {
            "benchmark_root": str(benchmark_root),
            "bddl_files": str(benchmark_root / "bddl_files"),
            "init_states": str(benchmark_root / "init_files"),
            "datasets": str(args.dataset.resolve()),
            "assets": str(benchmark_root / "assets"),
        }
    )
)
