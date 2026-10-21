#!/usr/bin/env python3
"""Guarded short HTP0 model smoke with routing trace and memory telemetry."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
MODEL_DEFAULT = Path("/data/data/com.termux/files/home/models/Qwen3-30B-A3B-Q4_K_M.gguf")
OUT = Path(__file__).resolve().parent
LLAMA = ROOT / ".work-npu/build/bin/llama-completion"
INVENTORY_TOOL = OUT / "inspect-expert-tensors.py"
CPU_STOP_C = 70
CPU_STOP_CONSECUTIVE_SAMPLES = 2
HTP_STOP_C = 60


def update_cpu_over_limit_streak(cpu_max_c, streak):
    """Count consecutive one-second samples strictly above the CPU stop."""
    if cpu_max_c is not None and cpu_max_c > CPU_STOP_C:
        return streak + 1
    return 0


def htp_stop_reason(htp_max_c):
    if htp_max_c is not None and htp_max_c > HTP_STOP_C:
        return f"HTP above {HTP_STOP_C} C"
    return None


def mem_field(path, key):
    key = key.rstrip(":")
    try:
        return next(int(line.split()[1]) for line in Path(path).read_text().splitlines()
                    if line.startswith(key + ":"))
    except (OSError, StopIteration, ValueError):
        return None


def routed_op_evidence(output):
    ops = []
    for line in output.splitlines():
        match = re.search(r"HTP0 execute-op MUL_MAT_ID\|([^|]+)\|.*\|HTP0 x HTP0 x HTP0 -> HTP0\|", line)
        if match and "_exps.weight" in match.group(1):
            ops.append({"tensor": match.group(1), "line": line.strip()})
    families = {family: any(f"ffn_{family}_exps.weight" in item["tensor"]
                            for item in ops)
                for family in ("gate", "up", "down")}
    return ops, families


def battery():
    try:
        result = subprocess.run(["termux-battery-status"], capture_output=True,
                                text=True, timeout=5, check=True)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError):
        return None


def efficiency_cpu_list():
    """Return CPUs from the lowest-numbered cluster (the two A520 cores here)."""
    clusters = {}
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        try:
            cpu = int(path.name[3:])
            cluster = int((path / "topology/cluster_id").read_text().strip())
            clusters.setdefault(cluster, []).append(cpu)
        except (OSError, ValueError):
            continue
    if not clusters:
        return None
    return ",".join(str(cpu) for cpu in sorted(clusters[min(clusters)]))


def stop(child):
    if child.poll() is not None:
        return
    child.send_signal(signal.SIGCONT)
    child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--inventory", type=Path,
                        default=OUT / "expert-inventory.json")
    parser.add_argument("--mode", choices=("smoke", "compare"), default="smoke")
    parser.add_argument("--ngl", type=int, choices=(1, 2, 4, 6), default=2,
                        help="bounded HTP layer count; 6 includes five repeating layers plus output")
    parser.add_argument("--tokens", type=int, choices=range(1, 9), default=1,
                        help="maximum generated tokens for smoke")
    parser.add_argument("--threads", type=int, choices=(1, 2), default=1,
                        help="CPU threads for the first bounded load attempt")
    parser.add_argument("--compare-steps", type=int, choices=range(1, 9), default=1,
                        help="full-vocabulary logit vectors for CPU/HTP comparison")
    parser.add_argument("--filter-htp-mul-mat-id", action="store_true",
                        help="force routed expert MUL_MAT_ID ops off HTP for attribution")
    parser.add_argument("--htp-only-mul-mat-id", action="store_true",
                        help="allow only routed expert MUL_MAT_ID ops on HTP during comparison")
    parser.add_argument("--htp-attention-core", action="store_true",
                        help="allow HTP attention primitives (MUL_MAT/SOFT_MAX/ROPE/SET_ROWS/CONT) and metadata ops")
    parser.add_argument("--htp-mul-mat-only", action="store_true",
                        help="allow only dense MUL_MAT plus metadata/layout ops on HTP; keep KV/KQV on CPU")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    filter_modes = (args.filter_htp_mul_mat_id, args.htp_only_mul_mat_id,
                    args.htp_attention_core, args.htp_mul_mat_only)
    if sum(filter_modes) > 1:
        parser.error("HTP op-filter modes are mutually exclusive")
    output_dir = args.out.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = args.model.resolve()
    slow_cpu_affinity = efficiency_cpu_list()
    if not args.run:
        print(json.dumps({"dry_run": True, "model": str(model), "trace_slots": 64,
                          "requested_context": 64, "runtime_min_context": 256,
                          "batch": 1, "threads": args.threads,
                          "ngl": args.ngl, "tokens": args.tokens,
                          "compare_steps": args.compare_steps,
                          "filter_htp_mul_mat_id": args.filter_htp_mul_mat_id,
                          "htp_only_mul_mat_id": args.htp_only_mul_mat_id,
                          "htp_attention_core": args.htp_attention_core,
                          "htp_mul_mat_only": args.htp_mul_mat_only,
                          "cpu_affinity": slow_cpu_affinity, "nice": 10,
                          "admit_memavailable_mib": 5120,
                          "compare_cpu_admit_c": 48,
                          "compare_interpass_cooldown_s": 180,
                          "stop_memavailable_mib": 2048,
                          "watchdog_s": 900}, indent=2))
        return
    output_name = args.mode
    if any((output_dir / name).exists() for name in (output_name + ".json", output_name + ".log", "memory.jsonl")):
        raise SystemExit("Smoke evidence exists; preserve it and use a new report directory")
    if not model.is_file() or not args.inventory.is_file():
        raise SystemExit("Model or expert inventory is missing; no inference started")
    inventory = json.loads(args.inventory.read_text())
    if inventory.get("htp_expert_type_gate") != "pass" or inventory.get("architecture") != "qwen3moe":
        raise SystemExit("GGUF expert type/architecture gate failed; no inference started")
    free = mem_field("/proc/meminfo", "MemAvailable:")
    battery_data = battery()
    if free is None or free < 5 * 1024 * 1024:
        raise SystemExit("Preflight refused: less than 5 GiB MemAvailable")
    if (not battery_data or int(battery_data.get("percentage", 0)) < 40
            or battery_data.get("plugged") == "UNPLUGGED"):
        raise SystemExit("Preflight refused: require AC connected and battery >= 40 percent")

    thermal_file = ROOT / "scripts/moe-thermal-monitor.py"
    spec = importlib.util.spec_from_file_location("htp_thermal", thermal_file)
    thermal_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(thermal_module)
    sensors = thermal_module.zones()
    if not sensors:
        raise SystemExit("Preflight refused: thermal sensors are unavailable")
    initial_thermal = thermal_module.snapshot(sensors)
    cpu_admission_c = 48 if args.mode == "compare" else 55
    if (initial_thermal.get("cpu_max_c") is not None and initial_thermal["cpu_max_c"] > cpu_admission_c
            or initial_thermal.get("battery_c") is not None and initial_thermal["battery_c"] > 40
            or initial_thermal.get("htp_max_c") is not None and initial_thermal["htp_max_c"] > 50):
        raise SystemExit("Preflight refused: CPU/HTP/battery temperature above soft limit")
    if not slow_cpu_affinity:
        raise SystemExit("Preflight refused: could not identify the low-power CPU cluster")

    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("GGML_HEXAGON_", "GGML_CPU_MOE_CACHE_")) or key in (
                "LD_PRELOAD", "GGML_BACKEND_PATH"):
            env.pop(key, None)
    env.update(LD_LIBRARY_PATH=str(ROOT / ".work-npu/build/bin"),
               ADSP_LIBRARY_PATH=str(ROOT / ".work-npu/build/ggml/src/ggml-hexagon"),
               GGML_HEXAGON_PROFILE="0", LLAMA_MMAP_PREFETCH="0")
    env["GGML_HEXAGON_OPFUSION"] = "0"
    if args.mode == "smoke":
        # The trace hook handles plain MUL_MAT_ID nodes. Fusion would turn
        # neighboring routed operations into MUL_MAT_ID_NX and hide IDs.
        # Emit backend support/fusion diagnostics so a clean decode with no
        # routed-op trace can be distinguished from CPU fallback.
        env["GGML_HEXAGON_VERBOSE"] = "1"
        env["GGML_HEXAGON_MOE_CACHE_TRACE_SLOTS"] = "64"
        command = ["taskset", "-c", slow_cpu_affinity, "nice", "-n", "10",
                   str(LLAMA), "-m", str(model), "-dev", "HTP0", "-ngl", str(args.ngl),
                   "-fit", "off", "-lm", "mmap", "-c", "64", "-b", "1", "-ub", "1",
                   "-t", str(args.threads), "-fa", "off", "--no-warmup",
                   "--no-repack",
                   "-n", str(args.tokens), "--n-cpu-moe", "0",
                   "--seed", "1", "--temp", "0", "--repeat-penalty", "1.1",
                   "--repeat-last-n", "64", "--log-verbosity", "5", "-no-cnv",
                   "-p", "Hi"]
    else:
        if any(filter_modes):
            # Hexagon's regex rejects matching op descriptions. Permit
            # NONE and layout nodes for HTP-resident tensors. In the attention
            # mode, retain the core attention op families and fall back for
            # other compute ops.
            if args.htp_only_mul_mat_id:
                env["GGML_HEXAGON_OPFILTER"] = "^(?!(MUL_MAT_ID|NONE)$).*"
            elif args.htp_attention_core:
                env["GGML_HEXAGON_OPFILTER"] = (
                    "^(?!(MUL_MAT|SOFT_MAX|ROPE|SET_ROWS|CONT|NONE|RESHAPE|VIEW|PERMUTE|TRANSPOSE)$).*")
            elif args.htp_mul_mat_only:
                env["GGML_HEXAGON_OPFILTER"] = (
                    "^(?!(MUL_MAT|NONE|RESHAPE|VIEW|PERMUTE|TRANSPOSE)$).*")
            else:
                env["GGML_HEXAGON_OPFILTER"] = "^MUL_MAT_ID$"
            env["GGML_HEXAGON_VERBOSE"] = "1"
        if args.htp_only_mul_mat_id:
            env["GGML_HEXAGON_CPU_KV"] = "1"
        if args.htp_mul_mat_only:
            env["GGML_HEXAGON_CPU_KV"] = "1"
        command = ["taskset", "-c", slow_cpu_affinity, "nice", "-n", "10",
                   str(ROOT / ".work-npu/build/bin/compare-cpu-htp-logits"),
                   str(model), str(args.compare_steps)]
    sources = [LLAMA, INVENTORY_TOOL, args.inventory.resolve(),
               OUT / "run-guarded-smoke.py",
               OUT / "compare-cpu-htp-logits.cpp",
               ROOT / ".work-npu/llama.cpp/src/llama-model.cpp",
               ROOT / ".work-npu/llama.cpp/ggml/src/ggml-hexagon/ggml-hexagon.cpp",
               ROOT / ".work-npu/llama.cpp/ggml/src/ggml-hexagon/htp/moe-expert-cache-trace.h",
               ROOT / ".work-npu/build/bin/llama-completion",
               ROOT / ".work-npu/build/bin/compare-cpu-htp-logits",
               ROOT / ".work-npu/build/bin/libggml-hexagon.so",
               ROOT / ".work-npu/build/ggml/src/ggml-hexagon/libggml-htp-v75.so"]
    provenance = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sources}
    result = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "model": str(model), "model_bytes": model.stat().st_size,
              "inventory": inventory, "command": command,
              "trace_slots": 64, "htp_verbose": args.mode == "smoke",
              "cpu_admission_c": cpu_admission_c,
              "interpass_cooldown_s": 180 if args.mode == "compare" else 0,
              "htp_op_filter": ("^(?!(MUL_MAT_ID|NONE)$).*" if args.htp_only_mul_mat_id
                                 else "^(?!(MUL_MAT|SOFT_MAX|ROPE|SET_ROWS|CONT|NONE|RESHAPE|VIEW|PERMUTE|TRANSPOSE)$).*" if args.htp_attention_core
                                 else "^(?!(MUL_MAT|NONE|RESHAPE|VIEW|PERMUTE|TRANSPOSE)$).*" if args.htp_mul_mat_only
                                 else "^MUL_MAT_ID$" if args.filter_htp_mul_mat_id else None),
              "htp_expert_only_cpu_kv": args.htp_only_mul_mat_id,
              "htp_force_cpu_kv": args.htp_only_mul_mat_id or args.htp_mul_mat_only,
              "cpu_affinity": slow_cpu_affinity,
              "nice": 10, "preflight": {"mem_available_kib": free,
                  "battery": battery_data, "thermal": initial_thermal},
              "source_sha256": provenance}
    reason = None
    min_available = None
    peak_rss = 0
    peak_swap = 0
    thermal_rows = []
    cpu_over_limit_streak = 0
    memory_path = output_dir / "memory.jsonl"
    thermal_path = output_dir / "thermal.jsonl"
    with (output_dir / (output_name + ".log")).open("w") as log, memory_path.open("w") as memory, thermal_path.open("w") as tlog:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                 stderr=subprocess.STDOUT)
        started = time.monotonic()
        next_battery = 0.0
        last_battery = battery_data
        while child.poll() is None:
            now = time.monotonic()
            available = mem_field("/proc/meminfo", "MemAvailable:")
            rss = mem_field(f"/proc/{child.pid}/status", "VmRSS:")
            vm_swap = mem_field(f"/proc/{child.pid}/status", "VmSwap:")
            minimum = mem_field(f"/proc/{child.pid}/smaps_rollup", "Pss:")
            sample = {"utc": datetime.now(timezone.utc).isoformat(),
                      "monotonic_s": now, "mem_available_kib": available,
                      "rss_kib": rss, "vm_swap_kib": vm_swap, "pss_kib": minimum}
            memory.write(json.dumps(sample) + "\n")
            memory.flush()
            if available is not None:
                min_available = min(min_available or available, available)
            peak_rss = max(peak_rss, rss or 0)
            peak_swap = max(peak_swap, vm_swap or 0)
            thermal = thermal_module.snapshot(sensors)
            thermal["utc"] = datetime.now(timezone.utc).isoformat()
            thermal_rows.append(thermal)
            tlog.write(json.dumps(thermal) + "\n")
            tlog.flush()
            if now >= next_battery:
                last_battery = battery() or last_battery
                next_battery = now + 10
            cpu = thermal.get("cpu_max_c")
            cpu_over_limit_streak = update_cpu_over_limit_streak(
                cpu, cpu_over_limit_streak)
            batt_temp = thermal.get("battery_c")
            htp = thermal.get("htp_max_c")
            if available is not None and available < 2 * 1024 * 1024:
                reason = "MemAvailable below 2048 MiB retained-headroom floor"
            elif int(last_battery.get("percentage", 0)) < 45:
                reason = "battery below 45 percent"
            elif cpu_over_limit_streak >= CPU_STOP_CONSECUTIVE_SAMPLES:
                reason = f"CPU above {CPU_STOP_C} C for {CPU_STOP_CONSECUTIVE_SAMPLES} consecutive samples"
            elif batt_temp is not None and batt_temp > 42:
                reason = "battery above 42 C"
            elif htp_stop_reason(htp):
                reason = htp_stop_reason(htp)
            elif now - started > 900:
                reason = "900-second watchdog"
            if reason:
                stop(child)
                break
            time.sleep(1)
        rc = child.wait()
    output = (output_dir / (output_name + ".log")).read_text(errors="replace")
    trace = re.search(r"HEX_MOE_CACHE_TRACE slots=(\d+) layers=(\d+) batches=(\d+) requests=(\d+) hits=(\d+) misses=(\d+) evictions=(\d+) fallback_batches=(\d+) invalid_batches=(\d+).*weights_cached=(\d+)", output)
    trace_record = ({"slots": int(trace.group(1)), "layers": int(trace.group(2)),
                     "batches": int(trace.group(3)), "requests": int(trace.group(4)),
                     "hits": int(trace.group(5)), "misses": int(trace.group(6)),
                     "evictions": int(trace.group(7)),
                     "fallback_batches": int(trace.group(8)),
                     "invalid_batches": int(trace.group(9)),
                     "weights_cached": int(trace.group(10))} if trace else None)
    trace_valid = (trace and int(trace.group(2)) > 0 and int(trace.group(3)) > 0
                   and int(trace.group(4)) > 0 and int(trace.group(9)) == 0)
    # The experimental cache trace currently misses model graph execution even
    # when the backend queues routed operations. Preserve those backend-level
    # records as independent evidence and require all three expert projections.
    routed_ops, routed_families = routed_op_evidence(output)
    routed_ops_valid = all(routed_families.values())
    comparison = None
    comparison_steps = []
    htp_expert_ops_seen = [line.strip() for line in output.splitlines()
                           if "HTP0 execute-op MUL_MAT_ID|" in line
                           and "_exps.weight" in line]
    htp_ops_seen = [line.strip() for line in output.splitlines()
                    if "HTP0 execute-op " in line]
    if args.mode == "compare":
        for line in output.splitlines():
            step_match = re.search(
                r"MODEL_LOGIT_METRIC step=(\d+)/(\d+) top1_agree=(\d+) kl=([\d.eE+-]+) rmse=([\d.eE+-]+) max_abs=([\d.eE+-]+)",
                line)
            if step_match:
                comparison_steps.append({"step": int(step_match.group(1)),
                                         "steps": int(step_match.group(2)),
                                         "top1_agree": bool(int(step_match.group(3))),
                                         "kl": float(step_match.group(4)),
                                         "rmse": float(step_match.group(5)),
                                         "max_abs": float(step_match.group(6))})
            try:
                row = json.loads(line)
                if isinstance(row, dict) and "all_finite" in row:
                    comparison = row
            except json.JSONDecodeError:
                pass
    mode_ok = ((trace_valid or routed_ops_valid) if args.mode == "smoke" else
               bool(comparison and comparison.get("all_finite")
                    and comparison.get("vocab_vectors") == comparison.get("steps")
                    and comparison.get("top1_agreements") == comparison.get("steps")
                    and comparison.get("mean_kl", float("inf")) <= 0.001))
    result.update({"finished_utc": datetime.now(timezone.utc).isoformat(),
                   "returncode": rc, "stop_reason": reason,
                   "cpu_stop_c": CPU_STOP_C,
                   "cpu_stop_consecutive_samples": CPU_STOP_CONSECUTIVE_SAMPLES,
                   "htp_stop_c": HTP_STOP_C,
                   "minimum_available_kib": min_available,
                   "peak_rss_kib": peak_rss, "peak_vm_swap_kib": peak_swap,
                   "final_battery": last_battery,
                   "trace_record": trace_record,
                   "routed_htp_ops": routed_ops,
                   "routed_htp_op_families": routed_families,
                   "htp_expert_ops_seen": htp_expert_ops_seen,
                   "htp_ops_seen": htp_ops_seen,
                   "filter_htp_mul_mat_id": args.filter_htp_mul_mat_id,
                   "htp_only_mul_mat_id": args.htp_only_mul_mat_id,
                   "htp_attention_core": args.htp_attention_core,
                   "htp_mul_mat_only": args.htp_mul_mat_only,
                   "htp_force_cpu_kv": args.htp_only_mul_mat_id or args.htp_mul_mat_only,
                   "htp_expert_only_cpu_kv": args.htp_only_mul_mat_id,
                   "comparison": comparison,
                   "comparison_steps": comparison_steps,
                   "thermal_sample_count": len(thermal_rows),
                   "status": "passed" if rc == 0 and reason is None and mode_ok else "failed"})
    (output_dir / (output_name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
