"""Run the project in a fresh kernel and export its report figures.

LLM-assisted: GitHub Copilot GPT-6 Astra (2026-10-05), Level 4 substantial generation.
"""

import base64
import json
from pathlib import Path
import sys
import time

from jupyter_client import KernelManager


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "project1_runge.ipynb"
FIGURES = {
    5: ("ols_mse.png", "ols_r2.png", "ols_fits.png"),
    12: ("ridge_test_metrics.png",),
    16: ("bootstrap_bias_variance.png",),
    19: ("cross_validation.png",),
    29: ("sgd_batch_sizes.png",),
    31: ("final_cv.png", "nested_cv.png"),
}


def execute_cell(client, source, timeout=1800):
    message_id = client.execute(source, stop_on_error=True)
    deadline = time.monotonic() + timeout
    outputs = []
    execution_count = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Notebook cell exceeded its execution timeout.")
        message = client.get_iopub_msg(timeout=remaining)
        if message["parent_header"].get("msg_id") != message_id:
            continue
        kind, content = message["msg_type"], message["content"]
        if kind == "execute_input":
            execution_count = content["execution_count"]
        elif kind == "stream":
            outputs.append({"output_type": kind, "name": content["name"],
                            "text": content["text"].splitlines(keepends=True)})
            print(content["text"], end="", flush=True)
        elif kind in ("display_data", "execute_result"):
            output = {"output_type": kind, "data": content["data"],
                      "metadata": content["metadata"]}
            if kind == "execute_result":
                output["execution_count"] = content["execution_count"]
            outputs.append(output)
        elif kind == "error":
            raise RuntimeError("\n".join(content["traceback"]))
        elif kind == "status" and content["execution_state"] == "idle":
            break
    return outputs, execution_count


def main():
    notebook = json.loads(NOTEBOOK.read_text())
    manager = KernelManager(kernel_name="python3")
    # Use the invoking virtual environment, not a user-level kernelspec's Python.
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]
    manager.start_kernel(cwd=str(ROOT))
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
        execute_cell(client, "%matplotlib inline", timeout=60)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            print(f"\nExecuting cell {index}", flush=True)
            cell["outputs"], cell["execution_count"] = execute_cell(
                client, "".join(cell["source"])
            )
            cell["metadata"].pop("ExecuteTime", None)
            cell["metadata"].pop("execution", None)
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)

    figure_data = {}
    for index, names in FIGURES.items():
        images = [output["data"]["image/png"]
                  for output in notebook["cells"][index]["outputs"]
                  if "image/png" in output.get("data", {})]
        if len(images) < len(names):
            raise ValueError(f"Cell {index} is missing report figures.")
        for name, image in zip(names, images):
            figure_data[name] = base64.b64decode(image)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    figure_directory = ROOT / "report_figures"
    figure_directory.mkdir(exist_ok=True)
    for name, image in figure_data.items():
        (figure_directory / name).write_bytes(image)
    print("\nSaved freshly executed notebook and report figures.", flush=True)


if __name__ == "__main__":
    main()
