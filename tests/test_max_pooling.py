import unittest
from unittest import mock

import torch

from models import DiT, max_pool_token_groups


class MaxPoolingTest(unittest.TestCase):
    def test_selects_elementwise_maximum_from_each_adjacent_group(self):
        tokens = torch.tensor(
            [[
                [1.0, 9.0],
                [4.0, 2.0],
                [3.0, 8.0],
                [0.0, 7.0],
                [-1.0, 5.0],
                [6.0, 1.0],
                [2.0, 3.0],
                [4.0, 0.0],
            ]]
        )
        expected = torch.tensor([[[4.0, 9.0], [6.0, 5.0]]])

        actual = max_pool_token_groups(tokens)

        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_incomplete_groups(self):
        with self.assertRaisesRegex(ValueError, "divisible by group_size"):
            max_pool_token_groups(torch.zeros(1, 6, 2))

    def test_dit_has_no_learned_x2_cls_tokens(self):
        with mock.patch.object(DiT, "load_pretrained_vit_weights", autospec=True):
            model = DiT(
                input_size=4,
                patch_size=2,
                hidden_size=8,
                depth=1,
                num_heads=1,
            )

        self.assertNotIn("x2_cls_tokens", dict(model.named_parameters()))


if __name__ == "__main__":
    unittest.main()
