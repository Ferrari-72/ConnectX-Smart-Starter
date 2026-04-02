from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingPreset:
    name: str
    episodes: int
    eval_games: int
    opponents: str
    checkpoint_path: str
    latest_checkpoint_path: str
    best_checkpoint_path: str
    eval_interval: int
    eval_opponents: str
    hybrid_depth: int
    description: str


PRESETS: dict[str, TrainingPreset] = {
    "quick": TrainingPreset(
        name="quick",
        episodes=200,
        eval_games=3,
        opponents="random",
        checkpoint_path="checkpoints/quick_latest.pt",
        latest_checkpoint_path="checkpoints/quick_latest.pt",
        best_checkpoint_path="checkpoints/quick_best.pt",
        eval_interval=50,
        eval_opponents="mixed",
        hybrid_depth=2,
        description="Fast smoke run for validating the pipeline.",
    ),
    "midpack": TrainingPreset(
        name="midpack",
        episodes=3000,
        eval_games=12,
        opponents="mixed",
        checkpoint_path="checkpoints/midpack_latest.pt",
        latest_checkpoint_path="checkpoints/midpack_latest.pt",
        best_checkpoint_path="checkpoints/midpack_best.pt",
        eval_interval=500,
        eval_opponents="ladder",
        hybrid_depth=2,
        description="Best first serious run for a single RTX 4060.",
    ),
    "overnight": TrainingPreset(
        name="overnight",
        episodes=10000,
        eval_games=20,
        opponents="ladder",
        checkpoint_path="checkpoints/overnight_latest.pt",
        latest_checkpoint_path="checkpoints/overnight_latest.pt",
        best_checkpoint_path="checkpoints/overnight_best.pt",
        eval_interval=1000,
        eval_opponents="ladder",
        hybrid_depth=2,
        description="Longer training pass for comparing checkpoints the next day.",
    ),
}


def get_training_preset(name: str) -> TrainingPreset:
    try:
        return PRESETS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown training preset: {name}") from exc


def resolve_training_args(
    preset_name: str,
    episodes: int | None,
    eval_games: int | None,
    opponents: str | None,
) -> TrainingPreset:
    preset = get_training_preset(preset_name)
    return TrainingPreset(
        name=preset.name,
        episodes=episodes if episodes is not None else preset.episodes,
        eval_games=eval_games if eval_games is not None else preset.eval_games,
        opponents=opponents if opponents is not None else preset.opponents,
        checkpoint_path=preset.checkpoint_path,
        latest_checkpoint_path=preset.latest_checkpoint_path,
        best_checkpoint_path=preset.best_checkpoint_path,
        eval_interval=preset.eval_interval,
        eval_opponents=preset.eval_opponents,
        hybrid_depth=preset.hybrid_depth,
        description=preset.description,
    )
