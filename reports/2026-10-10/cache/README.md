# Bounded CPU expert cache prototype, October 10, 2026

An opt-in 32-slot cache now runs the real Qwen3.6-35B-A3B UD-IQ1_M model
through the existing CPU+HTP build. A full-model correctness gate passed:
all 248,320 vocabulary logits were bit-identical at all 16 evaluated
positions, including the initial nine-token prompt and 15 teacher-forced
decode evaluations. The cache allocated 1.09 GiB of expert payload and
performed 2,838 evictions. No experts are removed from the model.

## Where the cache actually runs

The current quantization's low-bit expert operations use **CPU fallback**.
Hexagon's `ggml_hexagon_supported_mul_mat_id` accepts Q4_0, Q4_1, Q8_0,
IQ4_NL, MXFP4, Q4_K, Q5_K and Q6_K; it rejects the IQ1/IQ2 types used by
this model. The prior load log reports a 10,248.11 MiB CPU_Mapped buffer
and 120 expert tensors unable to use CPU_REPACK. A reported count of
layers offloaded to HTP0 does not establish that every operation in those
layers runs on the NPU. Earlier references to this model's routed experts
running through the NPU should be read with this correction.

Consequently the smallest working cache belongs in the CPU backend's
`graph_compute` path. It does not require changing `build_moe_ffn` or the
Hexagon kernels. The existing NPU computation is retained. This does not
prove cache feasibility for a different quantization whose expert kernels
actually execute on Hexagon.

## Mechanism and bounds

Each canonical, host-resident, quantized `blk.N.ffn_*_exps.weight` tensor
gets up to 32 contiguous expert slots. Separate gate/up/down tensors have
separate slot tables; across the 40 target layers their total payload is
1,168,375,808 bytes (1.09 GiB). A 1,280 MiB per-CPU-backend byte limit also
bounds allocation across tensors. Metadata, graph workspace, model state,
other weights and runtime memory are additional. Multiple CPU backends or
contexts can each allocate their own cache. This bounds allocated payload;
it does not lock pages in RAM or prevent Android from swapping them.

Before an eligible expert operation, any preceding CPU graph nodes finish
synchronously so the router output is ready. The cache copies the original
router IDs, checks the complete selected set, and pins every selected
expert against eviction while filling missing slots from the unchanged
mmap-backed weights. Unselected slots are evicted by recency. Both strided
prefill IDs and ordinary decode IDs are supported.

A temporary operation uses compact weight metadata and a private remapped
ID tensor. It finishes synchronously before the temporary objects are
released. Original model tensors, original graph nodes, original router
IDs and routing probabilities are preserved. Later scale/probability
lookups therefore still address the original experts.

If the selected union for a batch exceeds the slot count, the byte budget
would be exceeded, a tensor's immutable identity differs, or a slot
allocation fails, computation falls back to the original full tensor.
Repacked tensors, non-host tensors and unsupported layouts also retain
normal computation. The feature is for inference with immutable model
weights; training and the separate prebuilt `graph_plan_compute` API are
not covered. No frequency grading, prefetching, persistence, or MTP
integration is implemented yet.

Counters count unique selected experts **per weight operation**, so a
layer's gate/up/down requests are counted separately. They are not
selection-weighted routing hit rates from the earlier mapping experiment.

## Correctness evidence

- Slot-manager tests cover pinning across mixed hits/misses, eviction,
  duplicate IDs, oversized batches, byte-budget fallback, invalid IDs,
  and tensor identity changes.
- Real CPU graph tests cover Q2_K, IQ2_XXS, IQ1_S and IQ1_M at one and six
  threads. Every output is bit-identical to cache-disabled computation,
  including after eviction and oversized-batch fallback. A downstream
  operation consumes original IDs to check that ID remapping stays local.
- Full-model testing uses the same loaded model with sequential baseline
  and cached contexts, HTP0, forced mmap, six threads, batch/microbatch
  128, Flash Attention off and no MTP. The cached context receives the
  exact baseline input tokens; every full-vocabulary logit is compared.
  All 16 comparisons passed with maximum absolute error zero.
- Fresh patch application, checksums, repeat-application detection and
  reverse-application checks passed. Installer scripts and pins are unchanged.

## Preliminary performance observation

This is **one baseline-then-cache pair**, not an order-controlled benchmark.
The cache runs second and may benefit from page-cache warmth. Timings
include prompt processing plus 15 decode evaluations and exclude model
loading/context setup. Both cache allocation and miss refills during
those evaluations are included. The cached trajectory is teacher-forced
from the baseline; this is a correctness experiment, not a sampled chat
throughput test.

| Measurement | Cache disabled | 32 slots |
|---|---:|---:|
| Evaluation time | 41.147 s | 24.003 s |
| Process storage reads during evaluations | 42.87 GiB | 10.54 GiB |

The observed reduction is 41.7% in evaluation time and 75.4% in storage
reads for this pair. It is encouraging, but the earlier profiling showed
large environmental variation; repeated trials with balanced order,
longer continuations, and controlled starting conditions are required
before claiming a reliable performance benefit.

Live cache counters: 1,920 requests, 9,069 unique-expert hits, 6,678 misses,
2,838 evictions, 2,032,656,384 copied bytes, and 75 oversized-batch
fallbacks. There were zero budget or identity fallbacks. Exit code was
zero and the watchdog did not stop the run. Minimum sampled available
memory was 4.79 GiB and peak sampled process RSS was 6.52 GiB across both
passes, including model loading.

## Reproduction and operation

The patch is experimental and is already applied to the local generated
checkout used for these results. For the tested pinned checkout:

```sh
python scripts/apply-moe-cache.py
cmake --build .work-npu/build --target ggml-cpu -j 2
```

The helper checks the revision, base source hash and patch hashes, and
refuses to overwrite conflicting CPU source changes. It permits the
existing unrelated NPU/handoff patches. It is not part of either installer;
the installer's existing clean-source checks may reject this additional
experimental modification on a later installer rerun.

Enable the cache explicitly for a run:

```sh
GGML_CPU_MOE_CACHE_SLOTS=32 GGML_CPU_MOE_CACHE_MIB=1280 \
  scripts/run-npu.sh /path/to/model.gguf -lm mmap -t 6 -n 32
```

Without `GGML_CPU_MOE_CACHE_SLOTS`, the backend follows the original path.
A value of `0` also disables it. Valid slot counts are 1–256; a valid byte
budget is 1–8192 MiB, with 1280 MiB used by default. The budget is an
allocation bound, not a guarantee that it fits a particular phone's free
RAM. Larger selected batches can use the full-model fallback.

Compile and rerun the correctness gate:

```sh
clang++ -std=c++17 -O2 -Wall -Wextra scripts/moe-cache-validate.cpp \
  -I.work-npu/llama.cpp/include -I.work-npu/llama.cpp/ggml/include \
  -L.work-npu/build/bin -Wl,-rpath,"$PWD/.work-npu/build/bin" \
  -lllama -lggml -lggml-base -o .work-npu/build/bin/moe-cache-validate
python reports/2026-10-10/cache/run-validation.py
```

The runner fixes the tested local model path and 16 steps, clears inherited
cache/Hexagon experiment settings, records process/memory status, and
terminates only its child below 400 MiB available memory or after 600 s.
Rerunning replaces its result artifacts. The gate selects greedy baseline
input tokens for reproducible numerical comparison; it does not test
sampling quality or MTP. Its second pass always uses a fresh 32-slot cache.

Unit and graph tests:

```sh
clang++ -std=c++17 -O2 -Wall -Wextra tests/test_moe_slot_cache.cpp \
  -o .work-npu/build/bin/test-moe-slot-cache
.work-npu/build/bin/test-moe-slot-cache
clang++ -std=c++17 -O2 -Wall -Wextra tests/test_moe_cache_graph.cpp \
  -I.work-npu/llama.cpp/ggml/include -L.work-npu/build/bin \
  -Wl,-rpath,"$PWD/.work-npu/build/bin" -lggml-cpu -lggml-base \
  -o .work-npu/build/bin/test-moe-cache-graph
LD_LIBRARY_PATH="$PWD/.work-npu/build/bin" .work-npu/build/bin/test-moe-cache-graph
```

To remove only this experimental patch, preserving earlier source patches:

```sh
git -C .work-npu/llama.cpp apply -R "$PWD/patches/experimental-cpu-moe-cache.patch"
cmake --build .work-npu/build --target ggml-cpu -j 2
```

Artifacts: `validation.json` (full-model gate and the paired measurements),
`cache-stats.json` (runtime counters), `run.json` (memory/process status),
`graph-test.txt` (quantized graph tests), and `provenance.json` (source
checksums). The generated verbose `validation.log` is gitignored.

Longer four-pass balanced-order measurements, bit-exact replay and accessible
thermal sensor readings are recorded in the [follow-up report](../cache-balanced/README.md).
