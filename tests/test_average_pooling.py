import unittest
from unittest import mock

import torch

from models import DiT, average_pool_token_groups


class AveragePoolingTest(unittest.TestCase):
    def test_averages_each_adjacent_group(self):
        tokens = torch.arange(16, dtype=torch.float32).reshape(1, 8, 2)
        expected = tokens.reshape(1, 2, 4, 2).mean(dim=2)

        actual = average_pool_token_groups(tokens)

        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_incomplete_groups(self):
        with self.assertRaisesRegex(ValueError, "divisible by group_size"):
            average_pool_token_groups(torch.zeros(1, 6, 2))

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
