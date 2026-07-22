# Average-Pooling Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace learned x2 CLS-token aggregation with post-ViT grouped average pooling.

**Architecture:** Keep all `4T` x2 patch tokens through conditioning and the ViT block, then use a small pure helper to reshape them into four-token groups and take their arithmetic mean. Remove the CLS parameter and all CLS-specific training/checkpoint bookkeeping.

**Tech Stack:** Python, PyTorch, unittest, timm

## Global Constraints

- Pool after the ViT block and output projection.
- Average each four adjacent x2 tokens into one main-stream token.
- Do not add a runtime pooling option or change fusion scheduling.
- New model state dictionaries must not contain `x2_cls_tokens`.

---

### Task 1: Grouped average-pooling behavior

**Files:**
- Create: `tests/test_average_pooling.py`
- Modify: `models.py`

**Interfaces:**
- Consumes: a tensor shaped `(N, 4T, D)`.
- Produces: `average_pool_token_groups(tokens: torch.Tensor, group_size: int = 4) -> torch.Tensor` shaped `(N, T, D)`.

- [ ] **Step 1: Write failing tests**

```python
import unittest
from unittest import mock

import torch

from models import DiT, average_pool_token_groups


class AveragePoolingTest(unittest.TestCase):
    def test_average_pool_token_groups_averages_each_adjacent_group(self):
        tokens = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        expected = tokens.reshape(1, 2, 4, 2).mean(dim=2)
        self.assertTrue(torch.equal(average_pool_token_groups(tokens), expected))

    def test_average_pool_token_groups_rejects_incomplete_groups(self):
        with self.assertRaisesRegex(ValueError, "divisible by group_size"):
            average_pool_token_groups(torch.zeros(1, 6, 2))
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_average_pooling -v`
Expected: collection fails because `average_pool_token_groups` is not defined.

- [ ] **Step 3: Add the minimal helper**

```python
def average_pool_token_groups(tokens, group_size=4):
    batch_size, sequence_length, hidden_size = tokens.shape
    if sequence_length % group_size != 0:
        raise ValueError(
            f"sequence length {sequence_length} must be divisible by group_size {group_size}"
        )
    return tokens.reshape(
        batch_size, sequence_length // group_size, group_size, hidden_size
    ).mean(dim=2)
```

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m unittest tests.test_average_pooling -v`
Expected: 2 passed.

### Task 2: Replace CLS aggregation in the model

**Files:**
- Modify: `tests/test_average_pooling.py`
- Modify: `models.py`

**Interfaces:**
- Consumes: processed x2 tokens shaped `(N, 4T, D)` in `DiT.forward`.
- Produces: pooled x2 tokens shaped `(N, T, D)` for existing fusion sites.

- [ ] **Step 1: Add a failing model-structure test**

```python
class AveragePoolingTest(unittest.TestCase):
    def test_dit_has_no_learned_x2_cls_tokens(self):
        with mock.patch.object(DiT, "load_pretrained_vit_weights", autospec=True):
            model = DiT(input_size=4, patch_size=2, hidden_size=8, depth=1, num_heads=1)
        self.assertNotIn("x2_cls_tokens", dict(model.named_parameters()))
```

- [ ] **Step 2: Run the structure test and verify RED**

Run: `python -m unittest tests.test_average_pooling.AveragePoolingTest.test_dit_has_no_learned_x2_cls_tokens -v`
Expected: FAIL because `x2_cls_tokens` remains a named parameter.

- [ ] **Step 3: Remove CLS creation, insertion, and extraction**

Delete the `self.x2_cls_tokens` parameter from `DiT.__init__`. In `DiT.forward`, leave `x2_embedder(x)` as `(N, 4T, D)` without inserting tokens. After `x2_vit_proj_out`, replace CLS extraction with:

```python
x2 = average_pool_token_groups(x2, group_size=4)
```

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m unittest tests.test_average_pooling -v`
Expected: 3 passed.

### Task 3: Remove CLS-specific fine-tuning bookkeeping

**Files:**
- Modify: `train_x2_finetune.py`

**Interfaces:**
- Consumes: the CLS-free `DiT` parameter set.
- Produces: training logs and partial checkpoints without CLS assertions or claims.

- [ ] **Step 1: Establish the failing static check**

Run: `rg -n "x2_cls_tokens|LEARNABLE CLS TOKENS|including x2_cls" train_x2_finetune.py`
Expected: matches in unfreezing, parameter reporting, and checkpoint logging.

- [ ] **Step 2: Remove every CLS-specific block**

Delete CLS unfreezing and parameter-count code, the CLS line in the breakdown, both checkpoint-verification blocks, and change the periodic save log to:

```python
logger.info(f"Saved checkpoint to {checkpoint_path} (model contains only trainable parameters)")
```

- [ ] **Step 3: Verify the static check is GREEN**

Run: `rg -n "x2_cls_tokens|LEARNABLE CLS TOKENS|including x2_cls" train_x2_finetune.py`
Expected: no matches and exit status 1.

### Task 4: Regression verification

**Files:**
- Verify: `models.py`
- Verify: `train_x2_finetune.py`
- Verify: `tests/test_average_pooling.py`

**Interfaces:**
- Consumes: completed average-pooling ablation.
- Produces: evidence that focused behavior, syntax, and existing relevant tests pass.

- [ ] **Step 1: Compile changed Python files**

Run: `python -m py_compile models.py train_x2_finetune.py tests/test_average_pooling.py`
Expected: exit status 0.

- [ ] **Step 2: Run focused pytest coverage**

Run: `python -m unittest tests.test_average_pooling tests.test_checkpoint_io -v`
Expected: all tests pass.

- [ ] **Step 3: Run existing standalone checks**

Run: `python test_freezing.py`
Expected: prints `SUCCESS: Freezing logic verified correctly.`

Run: `python test_skip_connection.py`
Expected: completes a forward pass with the expected output shape.

- [ ] **Step 4: Inspect the final diff**

Run: `git diff --check && git diff -- models.py train_x2_finetune.py tests/test_average_pooling.py`
Expected: no whitespace errors; diff contains only the average-pooling ablation and its tests.
