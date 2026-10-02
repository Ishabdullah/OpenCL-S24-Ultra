"""Require explicit driver/device or successful model-offload evidence."""
import json
from pathlib import Path
import re
import sys

def parse(mode, text):
    if mode == "device":
        if not re.search(r"sphal InitOpenCLDriver\(\) = 0\b", text):
            raise ValueError("driver initialization success not reported")
        match = re.search(r"^\s*GPUOpenCL:\s*(.*Adreno.*)$", text, re.M)
        if not match:
            raise ValueError("Adreno GPUOpenCL missing; CPU-only enumeration is not a pass")
        return {"driver_initialized": True, "device": match.group(1),
                "tested_device_family": "Adreno(TM) 750" in match.group(1), "model_offload_tested": False}
    if mode == "inference":
        layers = re.search(r"offloaded (\d+)/(\d+) layers to GPU", text)
        buffer = re.search(r"OpenCL model buffer size\s*=\s*([\d.]+) MiB", text)
        generation = re.search(r"(?<!prompt )eval time\s*=\s*[\d.]+ ms /\s*(\d+) runs", text)
        if not layers or int(layers[1]) == 0 or not buffer or float(buffer[1]) <= 0:
            raise ValueError("positive GPU layer offload and OpenCL weights not confirmed")
        if not generation or int(generation[1]) == 0:
            raise ValueError("no completed generation evaluation reported")
        return {"model_offload_tested": True, "offloaded_layers": int(layers[1]),
                "total_layers": int(layers[2]), "OpenCL_weights_MiB": float(buffer[1]),
                "generation_evaluations": int(generation[1]), "process_exit_code": 0}
    raise ValueError("unknown mode")

if __name__ == "__main__":
    try:
        result = parse(sys.argv[1], Path(sys.argv[2]).read_text())
        Path(sys.argv[3]).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as error:
        sys.exit("Verification failed: " + str(error))
