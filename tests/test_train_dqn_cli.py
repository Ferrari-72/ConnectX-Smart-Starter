from datetime import datetime
from pathlib import Path

from scripts.train_dqn import derive_best_checkpoint_path, resolve_run_checkpoint_paths


def test_explicit_checkpoint_path_keeps_existing_best_sibling_rule():
    latest_path, best_path = resolve_run_checkpoint_paths(
        preset_name="midpack",
        save_checkpoint=Path("checkpoints/custom_latest.pt"),
        now=datetime(2026, 4, 2, 1, 30, 0),
    )

    assert latest_path == Path("checkpoints/custom_latest.pt")
    assert best_path == Path("checkpoints/custom_latest_best.pt")


def test_default_checkpoint_paths_use_unique_run_directory():
    latest_path, best_path = resolve_run_checkpoint_paths(
        preset_name="midpack",
        save_checkpoint=None,
        now=datetime(2026, 4, 2, 1, 30, 0),
    )

    assert latest_path == Path("checkpoints/runs/midpack-20260402-013000/latest.pt")
    assert best_path == Path("checkpoints/runs/midpack-20260402-013000/best.pt")


def test_derive_best_checkpoint_path_changes_only_filename():
    latest_path = Path("checkpoints/runs/midpack-20260402-013000/latest.pt")

    assert derive_best_checkpoint_path(latest_path) == Path("checkpoints/runs/midpack-20260402-013000/latest_best.pt")
