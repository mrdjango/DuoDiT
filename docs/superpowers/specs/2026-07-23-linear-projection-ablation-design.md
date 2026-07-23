# Linear-Projection Ablation Design

## Goal

Replace DuoDiT's learned per-group CLS-token aggregation with a learned linear projection on the linear-projection ablation branch.

## Model behavior

The x2 embedder continues to produce `4T` fine-grained patch tokens. These tokens pass through the existing optional conditioning, projection, and pretrained ViT block without inserted CLS tokens. After projection back to the model hidden size, reshape the sequence from `(N, 4T, D)` to `(N, T, 4D)` and apply a shared `nn.Linear(4D, D)` layer named `x2_group_projection`. Its `(N, T, D)` output fuses with the main stream through the existing fusion schedule.

Post-ViT projection preserves fine-token attention and changes only the aggregation method. The model no longer defines or checkpoints `x2_cls_tokens`.

## Initialization, training, and checkpoints

Initialize `x2_group_projection` with Xavier-uniform weights and zero bias, matching other projection layers. Fine-tuning must unfreeze and report this layer. Partial training checkpoints include its weight and bias through the existing trainable-parameter selection. Remove CLS-specific unfreezing, counts, assertions, logging, and checkpoint verification.

Existing CLS-based checkpoints require non-strict loading or migration. New checkpoints contain `x2_group_projection` and do not contain `x2_cls_tokens`.

## Verification

Add deterministic unit tests proving exact flatten-then-linear behavior, correct `(N, 4T, D) → (N, T, D)` shape, rejection of incomplete groups, absence of the CLS parameter, and presence of the new projection parameters. Run syntax checks, all repository unit tests, the freezing check, and a full DiT-XL/2 forward pass.

## Scope

Do not add a runtime aggregation option, change the main DiT stream or fusion scheduling, or modify unrelated evaluation code.
