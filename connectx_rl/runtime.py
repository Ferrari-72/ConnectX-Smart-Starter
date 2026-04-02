from __future__ import annotations

import torch


def module_device(module: torch.nn.Module) -> str | None:
    try:
        return next(module.parameters()).device.type
    except StopIteration:
        return None


def resolve_device(preferred: str = "auto") -> str:
    if preferred == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available in this PyTorch install.")
        return "cuda"
    if preferred == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device preference: {preferred}")


def resolve_module_device(module: torch.nn.Module, preferred: str = "auto") -> str:
    if preferred == "auto":
        detected = module_device(module)
        if detected is not None:
            return detected
    return resolve_device(preferred)
