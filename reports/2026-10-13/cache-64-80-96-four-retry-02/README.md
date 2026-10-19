# Controlled four-thread cache capacity comparison (Completed)

This experiment successfully ran to completion on October 13, 2026, executing a balanced 6-pass sweep comparing **64 vs 80 vs 96 slots** at four CPU threads with verified inter-pass thermal cooldowns.

## Execution Summary

- **Execution order**: **64 → 80 → 96 → 96 → 80 → 64** slots (budgets: 2560 / 3200 / 3840 MiB)
- **Token length**: 64 output tokens per pass (1 prefill + 63 decode)
- **Verification**: All 6 passes returned exit 0; all 64 replayed vocabulary positions were bit-identical against the reference.
- **Cooling**: 5 interpass pauses recorded (>=180s each, with battery <=40°C, CPU <=55°C, HTP <=50°C targets met).
- **Memory safety**: Minimum MemAvailable was 1,859,064 KiB (~1.86 GiB); peak child RSS was 6,461,660 KiB; zero budget or identity fallbacks.

## Ranked Configurations

| Rank | Slots | Budget | Allocated Payload | Hits / Misses | Evictions | Pooled Decode tok/s | Mean Evaluation s | Storage Reads (Mean) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **80** | **3200 MiB** | 2,920,939,520 B (~2.72 GiB) | 48,594 / 18,279 | 8,679 | **1.184 tok/s** | **61.56 s** | 18.72 GiB |
| **2** | **96** | **3840 MiB** | 3,505,127,424 B (~3.26 GiB) | 50,313 / 16,833 | 5,337 | **0.964 tok/s** | **75.28 s** | 17.14 GiB |
| **3** | **64** | **2560 MiB** | 2,336,751,616 B (~2.18 GiB) | 45,384 / 20,232 | 12,552 | **0.788 tok/s** | **88.18 s** | 22.51 GiB |

*Note: Fastest single pass occurred in Pass 4 (96 slots) at **1.300 tok/s**, but across both repetitions, 80 slots yielded the highest sustained pooled throughput.*

### Balanced Pair Comparisons

- **Pass 1 (64 slots) vs Pass 2 (80 slots)**: 80 slots achieved **1.145× decode speedup** (1.120 vs 0.978 tok/s) and **1.144× evaluation speedup**.
- **Pass 6 (64 slots) vs Pass 5 (80 slots)**: 80 slots achieved **1.901× decode speedup** (1.255 vs 0.660 tok/s) and **1.750× evaluation speedup**.
- **Overall**: 80 slots delivered a **1.50× higher pooled decode rate** than 64 slots under identical cooling and 4-thread CPU settings.

## Artifacts on Disk

- `analysis.json`: Aggregated metrics, per-pass breakdown, balanced pairs, and ranked configurations.
- `benchmark.jsonl` & `benchmark.log`: Complete per-token and per-forward raw records.
- `cache-stats.json`: Requests, hits, misses, evictions, copied bytes, and fallback counters per pass.
- `cooldown.jsonl` & `cooldown-status.json`: 5 verified cooling intervals with temperature traces.
- `memory.jsonl`: Timestamped MemAvailable, RSS, VmSwap, and PSS samples.
- `thermal.jsonl`: Thermal zone telemetry throughout all passes.
- `provenance.json` & `run.json`: Execution environment, exact SHA256 hashes, and command parameters.
