# Linear-Projection Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace learned x2 CLS-token aggregation with a shared post-ViT `4D → D` projection.

**Architecture:** Keep all `4T` x2 tokens through the ViT and output adapter, flatten adjacent four-token groups, and apply `x2_group_projection`. Remove CLS parameters and make the new layer part of x2 fine-tuning and partial checkpoints.

**Tech Stack:** Python, PyTorch, unittest, timm

## Global Constraints

- Aggregate after the ViT block and output projection.
- Use one shared `nn.Linear(4 * hidden_size, hidden_size)` named `x2_group_projection`.
- Do not add a runtime aggregation option or alter fusion scheduling.
- New state dictionaries contain `x2_group_projection` and not `x2_cls_tokens`.

---

### Task 1: Flatten-and-project behavior

**Files:**
- Create: `tests/test_linear_projection.py`
- Modify: `models.py`

**Interfaces:**
- Consumes: `(N, 4T, D)` tokens and `nn.Linear(4D, D)`.
- Produces: `project_token_groups(tokens, projection, group_size=4)` shaped `(N, T, D)`.

- [ ] **Step 1: Write failing deterministic tests**

```python
import unittest
import torch
import torch.nn as nn
from models import project_token_groups

class LinearProjectionTest(unittest.TestCase):
    def test_flattens_each_group_before_projection(self):
        tokens = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        projection = nn.Linear(8, 2, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]]))
        expected = torch.tensor([[[0.0, 7.0], [8.0, 15.0]]])
        self.assertTrue(torch.equal(project_token_groups(tokens, projection), expected))

    def test_rejects_incomplete_groups(self):
        with self.assertRaisesRegex(ValueError, "divisible by group_size"):
            project_token_groups(torch.zeros(1, 6, 2), nn.Linear(8, 2))
```

- [ ] **Step 2: Run `python -m unittest tests.test_linear_projection -v`; expect import failure for the absent helper.**

- [ ] **Step 3: Implement the helper**

```python
def project_token_groups(tokens, projection, group_size=4):
    batch_size, sequence_length, hidden_size = tokens.shape
    if sequence_length % group_size != 0:
        raise ValueError(f"sequence length {sequence_length} must be divisible by group_size {group_size}")
    grouped = tokens.reshape(batch_size, sequence_length // group_size, group_size * hidden_size)
    return projection(grouped)
```

- [ ] **Step 4: Run the two helper tests; expect both to pass.**

### Task 2: Replace CLS aggregation in DiT

**Files:**
- Modify: `tests/test_linear_projection.py`
- Modify: `models.py`

**Interfaces:**
- Produces: `DiT.x2_group_projection: nn.Linear(4D, D)` and CLS-free `(N, T, D)` x2 fusion tokens.

- [ ] **Step 1: Add a failing structure test**

```python
from unittest import mock
from models import DiT

def test_dit_uses_group_projection_instead_of_cls_tokens(self):
    with mock.patch.object(DiT, "load_pretrained_vit_weights", autospec=True):
        model = DiT(input_size=4, patch_size=2, hidden_size=8, depth=1, num_heads=1)
    parameters = dict(model.named_parameters())
    self.assertNotIn("x2_cls_tokens", parameters)
    self.assertEqual(model.x2_group_projection.in_features, 32)
    self.assertEqual(model.x2_group_projection.out_features, 8)
```

- [ ] **Step 2: Run the structure test; expect failure because CLS remains and the projection is absent.**
- [ ] **Step 3: Delete CLS creation/insertion/extraction, define `x2_group_projection`, and call `project_token_groups` after `x2_vit_proj_out`.**
- [ ] **Step 4: Run all linear-projection tests; expect three passes.**

### Task 3: Fine-tuning and checkpoint bookkeeping

**Files:**
- Modify: `train_x2_finetune.py`
- Modify: `tests/test_linear_projection.py`

**Interfaces:**
- Produces: trainable `x2_group_projection.weight` and `.bias` included by existing trainable-key checkpoint selection.

- [ ] **Step 1: Add a test that freezes all small-model parameters, unfreezes `x2_group_projection`, and asserts both parameters require gradients.**
- [ ] **Step 2: Run it before training-script changes to establish the expected projection interface.**
- [ ] **Step 3: Replace CLS unfreezing/count/log/checkpoint blocks with `x2_group_projection` unfreezing and parameter reporting; checkpoint selection remains automatic.**
- [ ] **Step 4: Verify no `x2_cls_tokens` references remain in `models.py` or `train_x2_finetune.py`.**

### Task 4: Regression verification

**Files:**
- Verify: `models.py`
- Verify: `train_x2_finetune.py`
- Verify: `tests/test_linear_projection.py`

**Interfaces:**
- Produces: evidence that the ablation is syntactically valid and shape-compatible.

- [ ] **Step 1: Run `python -m py_compile models.py train_x2_finetune.py tests/test_linear_projection.py`; expect exit 0.**
- [ ] **Step 2: Run `python -m unittest discover -s tests -v`; expect zero failures.**
- [ ] **Step 3: Run `python test_freezing.py`; expect the freezing success message.**
- [ ] **Step 4: Run `python test_skip_connection.py`; expect x2 and x shapes `(2, 256, 1152)` and output `(2, 8, 32, 32)`.**
- [ ] **Step 5: Run `git diff --check` and inspect the final diff.**
