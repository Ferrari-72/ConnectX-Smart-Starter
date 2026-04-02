from connectx_rl.training_presets import get_training_preset, resolve_training_args


def test_midpack_preset_uses_mixed_opponents_and_checkpointing_defaults():
    preset = get_training_preset("midpack")

    assert preset.name == "midpack"
    assert preset.opponents == "mixed"
    assert preset.episodes >= 1000
    assert preset.eval_games >= 10
    assert preset.eval_interval >= 100
    assert preset.best_checkpoint_path.endswith(".pt")
    assert preset.latest_checkpoint_path.endswith(".pt")
    assert preset.eval_opponents in {"mixed", "ladder"}
    assert preset.hybrid_depth >= 2


def test_explicit_cli_values_override_preset():
    resolved = resolve_training_args(
        preset_name="quick",
        episodes=123,
        eval_games=7,
        opponents="random",
    )

    assert resolved.episodes == 123
    assert resolved.eval_games == 7
    assert resolved.opponents == "random"
    assert resolved.eval_interval == get_training_preset("quick").eval_interval


def test_quick_preset_stays_small_for_smoke_runs():
    preset = get_training_preset("quick")

    assert preset.episodes < get_training_preset("midpack").episodes
    assert preset.eval_games <= 5
    assert preset.eval_interval <= preset.episodes
