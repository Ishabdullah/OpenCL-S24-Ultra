# Balanced CPU/GPU/HTP placement attempt

## Requested comparison

Qwen3-30B-A3B Q4_K_M has 48 repeating transformer blocks and one output layer. The intended contiguous map keeps blocks 0–15 on CPU, 16–31 on Adreno OpenCL, and 32–47 on Hexagon HTP. Three runs vary only the output layer:

| Run | CPU | GPU | HTP | Output layer |
| --- | ---: | ---: | ---: | --- |
| A | 17 | 16 | 16 | CPU |
| B | 16 | 17 | 16 | GPU |
| C | 16 | 16 | 17 | HTP |

Each placement is intended to use a two-pass deterministic replay at 64 expert-cache slots and one CPU thread. A 2-token load/replay smoke precedes any longer speed measurement. Temperature readings are used only for the established cooldown waits; memory availability remains monitored during loading and inference.

## Implementation

The normal llama.cpp tail placement cannot express this CPU/GPU/HTP interleave. The opt-in `MOE_BENCH_LAYER_MAP` loader extension assigns each of the 48 block layers and output layer individually. Its source patch is [experimental-explicit-layer-map.patch](../../../../patches/experimental-explicit-layer-map.patch). The benchmark's `--triad-output {cpu,gpu,npu}` option emits the three maps and verifies the actual per-layer backend assignment. When that environment variable is absent, standard placement is unchanged.

## Device attempt

Run A (CPU output) was attempted twice with 64 slots, one thread, and two tokens:

- At a 3.5 GiB `MemAvailable` floor, the memory monitor stopped the process during late tensor loading. The minimum recorded sample was 3,270,232 KiB. No decode or replay began. See [the first attempt](../attempt-20261020-triad-balanced-output-cpu-smoke/).
- At the established 2.0 GiB floor, the monitor stopped the process while the available-memory sample fell to 1,572,740 KiB. The Termux session remained responsive. No benchmark config row, decode, or replay was produced. See [the guarded retry](../attempt-20261020-triad-balanced-output-cpu-floor2-smoke/).

The guarded retry logged all 49 assignments before stopping: 17 CPU layers, 16 GPU layers, and 16 HTP layers, with output layer 48 on CPU. The map itself is verified in that attempt's `placement.json`; model-load completion and inference were not achieved. After the process stopped, available memory recovered to about 7.1 GiB and no benchmark process remained.

The 16 HTP repeating-layer placement therefore did not pass the safe-load smoke. Runs B and C were not started: each retains the same 16 HTP repeating layers, so moving only the output layer cannot address the observed load-time memory pressure. No inference-speed comparison or claim about which output backend is faster can be made from these attempts.

## Verification

The runner supports all three dry-run maps, records explicit assignments in loader logs, and now records cleanly when a guarded load ends before the benchmark emits its config row. The repository unit suite and `git diff --check` are run after these changes.
