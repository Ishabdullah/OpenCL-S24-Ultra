#!/usr/bin/env python3
"""Bounded HTP MUL_MAT_ID support and correctness checks for Snapdragon HTP0."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BIN = ROOT / ".work-npu/build/bin/test-backend-ops"
LIBS = ROOT / ".work-npu/build/bin"
ADSP = ROOT / ".work-npu/build/ggml/src/ggml-hexagon"
SUPPORTED = "type_a=(q4_0|q4_1|q8_0|iq4_nl|mxfp4|q4_K|q5_K|q6_K).*n_mats=4,n_used=2.*m=64,n=16"
SUPPORTED_TYPES = r"(?=.*type_a=q4_0.*SUPPORTED)(?=.*type_a=q4_1.*SUPPORTED)(?=.*type_a=q8_0.*SUPPORTED)(?=.*type_a=iq4_nl.*SUPPORTED)(?=.*type_a=mxfp4.*SUPPORTED)(?=.*type_a=q4_K.*SUPPORTED)(?=.*type_a=q5_K.*SUPPORTED)(?=.*type_a=q6_K.*SUPPORTED)"
REJECTED = "type_a=(iq1_s|iq1_m|iq2_xxs|iq2_xs|iq2_s).*n_mats=4,n_used=2.*m=64,n=16,k=768"


def read_int(path, prefix=None):
    try:
        value = Path(path).read_text().strip()
        if prefix:
            value = next(line.split()[1] for line in value.splitlines()
                         if line.startswith(prefix + ":"))
        return int(value)
    except (OSError, StopIteration, ValueError):
        return None


def temps():
    out = {}
    for path in Path("/sys/class/thermal").glob("thermal_zone*/type"):
        try:
            name = path.read_text().strip().lower()
            raw = int((path.parent / "temp").read_text().strip())
            if any(term in name for term in ("cpu", "battery", "htp", "npu")):
                out[name] = raw / 1000 if abs(raw) > 1000 else float(raw)
        except (OSError, ValueError):
            pass
    return out


def run_case(name, args, expected, env):
    started = time.monotonic()
    with (OUT / (name + ".log")).open("w") as log:
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, text=True)
        peak_rss = 0
        min_available = None
        while child.poll() is None:
            rss = read_int(f"/proc/{child.pid}/status", "VmRSS")
            available = read_int("/proc/meminfo", "MemAvailable")
            peak_rss = max(peak_rss, rss or 0)
            if available is not None:
                min_available = min(min_available or available, available)
            if min_available is not None and min_available < 768 * 1024:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise RuntimeError(f"{name}: stopped below 768 MiB MemAvailable")
            if time.monotonic() - started > 300:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise RuntimeError(f"{name}: 300-second watchdog")
            time.sleep(0.25)
    output = (OUT / (name + ".log")).read_text(errors="replace")
    if child.returncode != 0 or not re.search(expected, output, re.S):
        raise RuntimeError(f"{name}: rc={child.returncode}; expected output not found")
    return {"name": name, "argv": args, "returncode": child.returncode,
            "elapsed_s": round(time.monotonic() - started, 3),
            "peak_rss_kib": peak_rss, "minimum_available_kib": min_available,
            "log": name + ".log"}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="perform physical HTP0 checks")
    parser.add_argument("--out", type=Path, help="new evidence directory (must not contain prior run data)")
    args = parser.parse_args()
    if not args.run:
        print("dry-run: run with --run to perform HTP0 synthetic MUL_MAT_ID checks")
        return
    if args.out:
        OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if any((OUT / name).exists() for name in ("run.json", "support.log", "correctness.log")):
        raise SystemExit("Evidence exists; preserve it and use a new report directory")
    if not BIN.is_file():
        raise SystemExit(f"Missing backend test binary: {BIN}")

    available = read_int("/proc/meminfo", "MemAvailable")
    battery = None
    battery_data = None
    try:
        battery_proc = subprocess.run(["termux-battery-status"], capture_output=True,
                                      text=True, timeout=5, check=True)
        battery_data = json.loads(battery_proc.stdout)
        battery = int(battery_data["percentage"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, json.JSONDecodeError):
        battery = read_int("/sys/class/power_supply/battery/capacity")
    thermal = temps()
    cpu_max = max((v for k, v in thermal.items() if "cpu" in k), default=None)
    battery_temp = next((v for k, v in thermal.items() if "battery" in k), None)
    if available is None or available < 2 * 1024 * 1024:
        raise SystemExit("Preflight refused: MemAvailable below 2 GiB")
    if battery is None:
        raise SystemExit("Preflight refused: battery percentage unavailable")
    if battery is not None and battery < 40:
        raise SystemExit("Preflight refused: battery below 40 percent")
    if cpu_max is not None and cpu_max > 60:
        raise SystemExit("Preflight refused: CPU sensor above 60 C")
    if battery_temp is not None and battery_temp > 42:
        raise SystemExit("Preflight refused: battery sensor above 42 C")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(LIBS)
    env["ADSP_LIBRARY_PATH"] = str(ADSP)
    env.pop("GGML_HEXAGON_PROFILE", None)
    base = [str(BIN)]
    started = datetime.now(timezone.utc).isoformat()
    sources = ["reports/2026-10-19/htp-expert-ops/run-htp-expert-ops.py",
               ".work-npu/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp",
               ".work-npu/llama.cpp/ggml/src/ggml-hexagon/htp/moe-expert-cache-trace.h",
               ".work-npu/build/bin/test-backend-ops",
               ".work-npu/build/bin/libggml-hexagon.so",
               ".work-npu/build/ggml/src/ggml-hexagon/libggml-htp-v75.so"]
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
              for name in sources}
    result = {"started_utc": started, "device": "Snapdragon 8 Gen 3 / HTP0 v75",
              "preflight": {"battery_percent": battery, "battery_status": battery_data,
                            "thermal_c": thermal,
                            "mem_available_kib": available},
              "source_sha256": hashes, "checks": []}
    try:
        result["checks"].append(run_case("support", base + ["support", "-b", "HTP0",
            "-o", "MUL_MAT_ID", "-p", SUPPORTED], SUPPORTED_TYPES, env))
        result["checks"].append(run_case("correctness", base + ["test", "-b", "HTP0",
            "-o", "MUL_MAT_ID", "-p", SUPPORTED], r"8/8 tests passed", env))
        result["checks"].append(run_case("iq_rejection", base + ["support", "-b", "HTP0",
            "-o", "MUL_MAT_ID", "-p", REJECTED],
            r"(?=.*type_a=iq1_s.*NOT SUPPORTED)(?=.*type_a=iq1_m.*NOT SUPPORTED)(?=.*type_a=iq2_xxs.*NOT SUPPORTED)(?=.*type_a=iq2_xs.*NOT SUPPORTED)(?=.*type_a=iq2_s.*NOT SUPPORTED)",
            env))
        result["status"] = "passed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
        raise
    finally:
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["postflight"] = {"mem_available_kib": read_int("/proc/meminfo", "MemAvailable"),
                                "thermal_c": temps()}
        (OUT / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
