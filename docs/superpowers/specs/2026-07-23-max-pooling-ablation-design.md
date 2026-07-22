# Max-Pooling Ablation Design

## Goal

Replace DuoDiT's learned per-group CLS-token aggregation with parameter-free max pooling on the max-pooling ablation branch.

## Model behavior

The x2 embedder continues to produce `4T` fine-grained patch tokens. These tokens pass through the existing optional conditioning, projection, and pretrained ViT block without inserted CLS tokens. After projection back to the model hidden size, reshape the sequence from `(N, 4T, D)` to `(N, T, 4, D)` and take the element-wise maximum across each four-token group, producing `(N, T, D)` for fusion with the main stream.

Pooling after the ViT block preserves fine-token attention and isolates the aggregation strategy. The model no longer defines or checkpoints `x2_cls_tokens`.

## Training and checkpoint behavior

Remove CLS-specific unfreezing, parameter counts, assertions, logging, and checkpoint verification from `train_x2_finetune.py`. Existing CLS-based checkpoints require non-strict loading or migration; new checkpoints do not contain `x2_cls_tokens`.

## Verification

Add deterministic unit tests proving that grouped max pooling selects the element-wise maximum from each four-token group and rejects incomplete groups. Assert that the CLS parameter is absent. Run syntax checks, all repository unit tests, the existing freezing check, and a full DiT-XL/2 forward pass.

## Scope

Do not add a runtime pooling option, change the main DiT stream or fusion scheduling, or modify unrelated evaluation code.
