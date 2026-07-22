# Max-Pooling Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace learned x2 CLS-token aggregation with post-ViT grouped max pooling.

**Architecture:** Keep all `4T` x2 patch tokens through conditioning and the ViT block, then reshape them into four-token groups and take the element-wise maximum. Remove the CLS parameter and all CLS-specific training/checkpoint bookkeeping.

**Tech Stack:** Python, PyTorch, unittest, timm

## Global Constraints

- Pool after the ViT block and output projection.
- Max-pool each four adjacent x2 tokens into one main-stream token.
- Do not add a runtime pooling option or change fusion scheduling.
- New model state dictionaries must not contain `x2_cls_tokens`.
- Do not create implementation commits.

---

### Task 1: Grouped max-pooling behavior

**Files:**
- Create: `tests/test_max_pooling.py`
- Modify: `models.py`

**Interfaces:**
- Consumes: a tensor shaped `(N, 4T, D)`.
- Produces: `max_pool_token_groups(tokens: torch.Tensor, group_size: int = 4) -> torch.Tensor` shaped `(N, T, D)`.

- [ ] **Step 1: Write failing tests** defining deterministic per-feature maxima and incomplete-group rejection with `unittest`.
- [ ] **Step 2: Run `python -m unittest tests.test_max_pooling -v`; expect import failure because the helper is absent.**
- [ ] **Step 3: Implement `max_pool_token_groups` by validating divisibility, reshaping to `(N, T, group_size, D)`, and returning `.amax(dim=2)`.**
- [ ] **Step 4: Run the two helper tests; expect both to pass.**

### Task 2: Replace CLS aggregation

**Files:**
- Modify: `tests/test_max_pooling.py`
- Modify: `models.py`

**Interfaces:**
- Consumes: processed x2 tokens shaped `(N, 4T, D)`.
- Produces: pooled x2 tokens shaped `(N, T, D)` for existing fusion sites.

- [ ] **Step 1: Add a test constructing a small `DiT` with pretrained loading mocked and asserting `x2_cls_tokens` is absent.**
- [ ] **Step 2: Run that test; expect failure because the parameter exists.**
- [ ] **Step 3: Delete CLS creation/insertion/extraction and call `max_pool_token_groups(x2, group_size=4)` after output projection.**
- [ ] **Step 4: Run all max-pooling tests; expect three passes.**

### Task 3: Fine-tuning cleanup and regression verification

**Files:**
- Modify: `train_x2_finetune.py`
- Verify: `models.py`
- Verify: `tests/test_max_pooling.py`

**Interfaces:**
- Consumes: CLS-free model parameters.
- Produces: training/checkpoint behavior without CLS-specific assumptions.

- [ ] **Step 1: Remove CLS unfreezing, counts, assertions, checkpoint verification, and save-log claims.**
- [ ] **Step 2: Verify `rg -n "x2_cls_tokens|LEARNABLE CLS TOKENS|including x2_cls" train_x2_finetune.py` returns no matches.**
- [ ] **Step 3: Run `python -m py_compile models.py train_x2_finetune.py tests/test_max_pooling.py`.**
- [ ] **Step 4: Run `python -m unittest discover -s tests -v`, `python test_freezing.py`, and `python test_skip_connection.py`; expect zero failures and matching `(N, T, D)` stream shapes.**
- [ ] **Step 5: Run `git diff --check` and inspect the final diff; do not commit.**
