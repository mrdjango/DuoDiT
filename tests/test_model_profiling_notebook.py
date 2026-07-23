import ast
import json
import os
import unittest
from pathlib import Path

import torch


NOTEBOOK = Path(
    os.environ.get(
        "MODEL_PROFILING_NOTEBOOK",
        "/Users/mrmc/Documents/GitHub/DuoDiT/model_profiling.ipynb",
    )
)


class DuoDiTModelProfilingNotebookTest(unittest.TestCase):
    def test_notebook_profiles_duodit_x2_finetuning(self):
        notebook = json.loads(NOTEBOOK.read_text())
        code_cells = [
            cell for cell in notebook["cells"] if cell["cell_type"] == "code"
        ]
        code = "\n\n".join("".join(cell["source"]) for cell in code_cells)

        self.assertIn("def measure_forward_flops(", code)
        self.assertIn("class CudaEpochMemoryTracker:", code)
        self.assertIn("from models import DiT_models", code)
        self.assertIn('MODEL = "DiT-XL/2"', code)
        self.assertIn("X2_VIT_DEPTH = 1", code)
        self.assertIn('TRAINING_MODE = "x2_finetune"', code)
        self.assertIn("model.x2_vit_blocks.parameters()", code)
        self.assertIn("model.x2_vit_proj_in", code)
        self.assertIn("create_diffusion", code)
        self.assertIn("INCLUDE_EMA = True", code)
        self.assertIn("paper_gflops_per_sample", code)
        self.assertIn("def profile_cuda_inference(", code)
        self.assertIn("class NvidiaSmiSampler:", code)
        self.assertIn("INFERENCE_PROFILE_STEPS = 20", code)
        self.assertIn('"samples_per_second"', code)
        self.assertIn('"peak_allocated_gib"', code)
        self.assertIn('"mean_gpu_util_percent"', code)
        self.assertIn('"max_power_w"', code)
        self.assertTrue(
            all(
                cell.get("execution_count") is None and cell.get("outputs") == []
                for cell in code_cells
            )
        )
        for cell in code_cells:
            ast.parse("".join(cell["source"]))

    def test_flop_helper_runs_for_a_cpu_model(self):
        notebook = json.loads(NOTEBOOK.read_text())
        utility_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "def measure_forward_flops(" in "".join(cell["source"])
        )
        namespace = {}
        exec("import torch\n" + utility_source, namespace)

        model = torch.nn.Linear(4, 3, bias=False)
        inputs = torch.ones(2, 4)
        stats = namespace["measure_forward_flops"](
            lambda: model(inputs),
            batch_size=2,
        )

        self.assertGreater(stats["hardware_gflops_batch"], 0)
        self.assertAlmostEqual(
            stats["paper_gflops_per_sample"] * 4,
            stats["hardware_gflops_batch"],
        )


if __name__ == "__main__":
    unittest.main()
