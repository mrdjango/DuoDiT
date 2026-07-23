import unittest
from unittest import mock

import torch
import torch.nn as nn

from models import DiT, project_token_groups


class LinearProjectionTest(unittest.TestCase):
    def test_flattens_each_group_before_projection(self):
        tokens = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        projection = nn.Linear(8, 2, bias=False)
        with torch.no_grad():
            projection.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    ]
                )
            )
        expected = torch.tensor([[[0.0, 7.0], [8.0, 15.0]]])

        actual = project_token_groups(tokens, projection)

        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_incomplete_groups(self):
        with self.assertRaisesRegex(ValueError, "divisible by group_size"):
            project_token_groups(torch.zeros(1, 6, 2), nn.Linear(8, 2))

    def test_dit_uses_group_projection_instead_of_cls_tokens(self):
        with mock.patch.object(DiT, "load_pretrained_vit_weights", autospec=True):
            model = DiT(
                input_size=4,
                patch_size=2,
                hidden_size=8,
                depth=1,
                num_heads=1,
            )

        parameters = dict(model.named_parameters())
        self.assertNotIn("x2_cls_tokens", parameters)
        self.assertEqual(model.x2_group_projection.in_features, 32)
        self.assertEqual(model.x2_group_projection.out_features, 8)

    def test_group_projection_parameters_can_be_unfrozen(self):
        with mock.patch.object(DiT, "load_pretrained_vit_weights", autospec=True):
            model = DiT(
                input_size=4,
                patch_size=2,
                hidden_size=8,
                depth=1,
                num_heads=1,
            )
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.x2_group_projection.parameters():
            parameter.requires_grad = True

        self.assertTrue(model.x2_group_projection.weight.requires_grad)
        self.assertTrue(model.x2_group_projection.bias.requires_grad)


if __name__ == "__main__":
    unittest.main()
