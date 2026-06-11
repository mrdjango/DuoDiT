import errno
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from checkpoint_io import (
    CheckpointLoadError,
    atomic_torch_save,
    get_resume_position,
    load_torch_checkpoint,
)


class CheckpointIoTest(unittest.TestCase):
    def test_resume_position_from_legacy_global_step(self):
        self.assertEqual(get_resume_position({"step": 150_000}, 1000), (150_000, 150, 0))
        self.assertEqual(get_resume_position({"step": 150_123}, 1000), (150_123, 150, 123))

    def test_resume_position_normalizes_completed_epoch(self):
        checkpoint = {"step": 150_000, "epoch": 149, "step_in_epoch": 1000}
        self.assertEqual(get_resume_position(checkpoint, 1000), (150_000, 150, 0))

    def test_atomic_save_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            expected = {"ema": {"weight": torch.arange(4)}}

            atomic_torch_save(expected, path)
            actual = load_torch_checkpoint(path)

            self.assertTrue(torch.equal(actual["ema"]["weight"], expected["ema"]["weight"]))
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_failed_save_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            atomic_torch_save({"step": 1}, path)

            def fail_after_partial_write(_obj, temporary_path):
                Path(temporary_path).write_bytes(b"partial")
                raise OSError(errno.EIO, "simulated write failure")

            with mock.patch("checkpoint_io.torch.save", side_effect=fail_after_partial_write):
                with self.assertRaises(OSError):
                    atomic_torch_save({"step": 2}, path)

            self.assertEqual(load_torch_checkpoint(path)["step"], 1)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_truncated_checkpoint_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.pt"
            torch.save({"weight": torch.arange(1024)}, path)
            path.write_bytes(path.read_bytes()[:64])

            with self.assertRaisesRegex(CheckpointLoadError, "incomplete, corrupt"):
                load_torch_checkpoint(path)

    def test_input_output_error_has_storage_diagnostic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            path.write_bytes(b"placeholder")

            with mock.patch(
                "checkpoint_io.torch.load",
                side_effect=OSError(errno.EIO, "simulated read failure"),
            ):
                with self.assertRaisesRegex(CheckpointLoadError, "filesystem returned"):
                    load_torch_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
