"""Run sequential, matched llama-bench CPU/OpenCL tests; retain raw results."""
import json
from pathlib import Path
import subprocess
import sys

def run(binary, model, output, threads, prompt, generation, repetitions):
    common = [binary, "-m", model, "-fa", "off", "-p", prompt, "-n", generation,
              "-b", "128", "-ub", "128", "-t", threads, "-r", repetitions,
              "-ctk", "f16", "-ctv", "f16", "-o", "json"]
    results = {}
    print(f"Matched: threads={threads}, pp={prompt}, tg={generation}, repetitions={repetitions}, Flash Attention=off, profiling=off", flush=True)
    for name, options in [("cpu", ["-ngl", "0", "-dev", "none"]), ("gpu", ["-ngl", "99", "-dev", "GPUOpenCL"])]:
        command = common + options
        print("Running " + name.upper() + "...", flush=True)
        (output / (name + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
        with (output / (name + ".json")).open("w") as out, (output / (name + ".stderr")).open("w") as err:
            proc = subprocess.run(command, stdout=out, stderr=err)
        if proc.returncode:
            raise ValueError(f"{name} failed (exit {proc.returncode}); see {output / (name + '.stderr')}")
        rows = json.loads((output / (name + ".json")).read_text())
        if len(rows) != 2:
            raise ValueError("expected one prompt and one generation row per backend")
        results[name] = rows
    keys = ["build_commit", "model_type", "model_size", "model_n_params", "model_filename",
            "n_threads", "n_batch", "n_ubatch", "type_k", "type_v", "flash_attn", "n_prompt", "n_gen"]
    summary = []
    print("Test       CPU tok/s          OpenCL tok/s       OpenCL vs CPU")
    for cpu, gpu in zip(results["cpu"], results["gpu"]):
        if any(cpu[k] != gpu[k] for k in keys):
            raise ValueError("metadata differ; refusing unmatched comparison")
        if cpu["n_gpu_layers"] != 0 or cpu["devices"] != "none" or gpu["n_gpu_layers"] != 99 or gpu["devices"] != "GPUOpenCL":
            raise ValueError("backend selection differs from requested comparison")
        label = "pp" + str(cpu["n_prompt"]) if cpu["n_prompt"] else "tg" + str(cpu["n_gen"])
        change = 100 * (gpu["avg_ts"] / cpu["avg_ts"] - 1)
        summary.append({"test": label, "cpu_tok_s": cpu["avg_ts"], "cpu_stddev": cpu["stddev_ts"],
                        "OpenCL_tok_s": gpu["avg_ts"], "OpenCL_stddev": gpu["stddev_ts"], "OpenCL_vs_CPU_percent": change})
        print(f"{label:<10} {cpu['avg_ts']:>9.2f} +/- {cpu['stddev_ts']:.2f}   {gpu['avg_ts']:>9.2f} +/- {gpu['stddev_ts']:.2f}   {change:+.1f}%")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Results and commands: " + str(output))
    print("A negative percentage means OpenCL is slower. Mobile clocks/thermal state are not controlled.")

if __name__ == "__main__":
    try:
        run(sys.argv[1], sys.argv[2], Path(sys.argv[3]), *sys.argv[4:8])
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as error:
        sys.exit("Benchmark failed: " + str(error))
