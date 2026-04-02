# Training Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add balanced training governance for ConnectX with stronger opponent pools, periodic DQN and hybrid evaluation, and automatic best/latest checkpoint management.

**Architecture:** Preserve the existing `DQNTrainer` as the low-level optimization engine and add a separate orchestration layer that runs chunked training and evaluation. Expand presets so the CLI can select runtime budget and governance policy together.

**Tech Stack:** Python, PyTorch, pytest, argparse, dataclasses, pathlib

---

### Task 1: Add failing orchestration tests

**Files:**
- Create: `tests/test_training_campaign.py`
- Test: `tests/test_training_campaign.py`

- [ ] **Step 1: Write the failing test**

```python
def test_campaign_trains_in_chunks_and_tracks_best_checkpoint():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_campaign.py -v`
Expected: FAIL because `connectx_rl.training_campaign` does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def run_training_campaign(...):
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_training_campaign.py -v`
Expected: PASS

### Task 2: Expand presets and CLI inputs

**Files:**
- Modify: `connectx_rl/training_presets.py`
- Modify: `tests/test_training_presets.py`
- Modify: `scripts/train_dqn.py`

- [ ] **Step 1: Write the failing preset tests**

```python
def test_midpack_preset_exposes_eval_interval_and_best_checkpoint():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_training_presets.py -v`
Expected: FAIL because the new fields are missing.

- [ ] **Step 3: Implement preset fields and CLI wiring**

```python
@dataclass(frozen=True)
class TrainingPreset:
    ...
    eval_interval: int
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_training_presets.py -v`
Expected: PASS

### Task 3: Integrate campaign into training CLI

**Files:**
- Modify: `connectx_rl/dqn_trainer.py`
- Modify: `scripts/train_dqn.py`
- Modify: `README.md`
- Test: `tests/test_training_campaign.py`

- [ ] **Step 1: Extend checkpoint save path support and campaign reporting**

```python
def save_checkpoint(self, path, metadata=None):
    ...
```

- [ ] **Step 2: Route `train_dqn` through the campaign**

Run: `python -m pytest tests/test_training_campaign.py tests/test_training_presets.py -v`
Expected: PASS

- [ ] **Step 3: Update README commands and descriptions**

```markdown
python -m scripts.train_dqn --preset midpack --device auto
```

- [ ] **Step 4: Run broader verification**

Run: `python -m pytest`
Expected: PASS
