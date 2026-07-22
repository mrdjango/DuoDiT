# Average-Pooling Ablation Design

## Goal

Replace DuoDiT's learned per-group CLS-token aggregation with parameter-free average pooling for the current pooling ablation branch.

## Model behavior

The x2 embedder continues to produce `4T` fine-grained patch tokens. These tokens pass through the existing optional conditioning, projection, and pretrained ViT block without any inserted CLS tokens. After projection back to the model hidden size, the sequence is reshaped from `(N, 4T, D)` to `(N, T, 4, D)` and averaged across each four-token group, producing `(N, T, D)` for fusion with the main stream.

Pooling after the ViT block preserves fine-token attention and changes only the aggregation strategy. The model no longer defines or checkpoints `x2_cls_tokens`.

## Training and checkpoint behavior

Remove CLS-specific unfreezing, parameter counts, assertions, logging, and checkpoint verification from `train_x2_finetune.py`. The x2 ViT block and projection layers retain their existing trainability rules. Existing checkpoints containing `x2_cls_tokens` require non-strict loading or migration; new checkpoints do not contain that key.

## Verification

Add a focused unit test for grouped average pooling using deterministic token values. It must prove that each output token is the arithmetic mean of its corresponding four adjacent input tokens and that invalid sequence lengths are rejected clearly. Add a model-structure assertion that the CLS parameter is absent. Run the focused tests plus the existing freezing and skip-connection tests.

## Scope

This change does not add a runtime pooling option, alter the main DiT stream, change fusion scheduling, or modify unrelated evaluation code.
