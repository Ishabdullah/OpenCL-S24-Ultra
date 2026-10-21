# CPU thread sweep: partial results

Run date: 2026-10-20 UTC  
Model: Qwen3-30B-A3B Q4_K_M  
Backend: CPU only, 64 cache slots / 2560 MiB, 16 generated positions, 1 batch / 1 ubatch  
Inference thermal policy: cooldown-only; 180-second minimum cooldown between completed passes, with battery/CPU/HTP targets recorded in `cooldown.jsonl`.

| CPU threads | Decode rate (tok/s) | Output replay |
|---:|---:|---|
| 1 | 0.516244 | Reference |
| 2 | 0.083661 | 16/16 positions bit-identical |
| 3 | 0.086736 | 16/16 positions bit-identical |
| 4 | 0.072267 | 16/16 positions bit-identical |
| 5 | 0.065075 | 16/16 positions bit-identical |

The user requested stopping after the five-thread result. The runner was stopped during its post-pass-5 cooldown, before pass 6; threads 6–8 were not tested. Raw timing, memory, and thermal data are in `benchmark.jsonl`, `memory.jsonl`, `thermal.jsonl`, and `cooldown.jsonl`.

In this run, one thread was the fastest measured configuration. The result is provisional: pass 1 read about 23.7 GB from storage according to the benchmark counters, while passes 2–5 each recorded about 120.9–163.6 GB. That large difference means the sweep includes a substantial change in storage activity/cache state across passes, so repeat the one-thread run and investigate the storage counters before treating the roughly 6x gap as a stable thread-scaling result. The earlier one-thread pilot pinned to CPUs 0–1 is not directly comparable; this sweep used the runner's all-allowed-CPU affinity policy.
