import torch

from models import DiT_S_2


def test_forward_shape_and_x2_branch_is_used():
    model = DiT_S_2(input_size=32, num_classes=10).eval()
    x = torch.randn(2, 4, 32, 32)
    t = torch.tensor([10, 500])
    y = torch.tensor([1, 2])
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == (2, 8, 32, 32)
    assert model.x2_cls_tokens.shape == (1, 256, model.hidden_size)
