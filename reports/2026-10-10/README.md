# Repeated-prompt expert mapping, October 10, 2026

This measures routing on the existing Qwen3.6-35B-A3B UD-IQ1_M model. It
leaves all expert weights and routing behavior intact. It is an experiment,
not an expert cache or a speedup demonstration.

The unfinished mapping probe and four older logs were recovered on resume.
The old logs show growing unions, but their terminal union JSON is
truncated, so they are not used as the quantitative evidence here.
The previous checkpoint files stopped at Phase G.

## Method

Eight trials use the identical raw completion prompt, `Write a Python
function to reverse a string:`, with 24 output tokens maximum and seeds
1000 through 1007. The model is loaded once; both attention and recurrent
memory are cleared between trials. Sampling uses top-k 40, temperature
0.8, and repeat penalty 1.1 over the last 64 tokens. Prompt tokens are
accepted into the sampler history. These settings differ from the old
unfinished probe, which had no repetition penalty.

Configuration: HTP0, all layers requested offloaded, forced mmap, six CPU
threads, batch/microbatch 128, Flash Attention disabled, MTP disabled.
This is a raw completion experiment, not a chat-template benchmark.

The callback captures the complete I32 expert-selection tensors, counts
selection occurrences per layer, and separates prefill from decode.
The final generated token is not evaluated, as in normal generation;
24 generated tokens therefore produce 23 decode forward evaluations.
Output text is JSON-escaped, including newlines. Each JSONL trial record
is flushed immediately. The analyzer checks all selection counts, layer
coverage, expert ID bounds, and cumulative union growth.

The first six trials define a frozen training map. The final two trials
measure selection-weighted coverage of that map, without adding their
experts to it. This tests held-out continuations of the same prompt only;
it does not establish coverage of longer output, other prompts, or a
conversation.

Tensor sizes come from the local GGUF header. Per-expert bytes are each
layer's gate/up/down expert-weight payload divided by its expert count.
The non-routed total includes every other tensor, including any unused
MTP tensors; it is a conservative payload accounting baseline, not an
exact required resident allocation. Weight payload estimates exclude
KV/recurrent state, compute buffers, alignment, allocator overhead,
application memory, and operating-system memory. A mapped subset is not
a safe replacement for the full model: unseen experts require a fallback.

The runner samples available memory and process RSS each second. It stops
its own child if available memory falls below 400 MiB or elapsed time
exceeds 900 seconds. Storage reads are `/proc/self/io` deltas per trial,
not direct per-expert I/O measurements. Instrumentation adds router
readback and host synchronization, so these timings are not a clean
baseline comparison.

## Reproduction

From the project root, with the existing NPU build and downloaded model:

```sh
clang++ -std=c++17 -O2 -Wall -Wextra scripts/moe-mapping-probe.cpp \
  -I.work-npu/llama.cpp/include -I.work-npu/llama.cpp/ggml/include \
  -L.work-npu/build/bin -Wl,-rpath,"$PWD/.work-npu/build/bin" \
  -lllama -lggml -lggml-base -o .work-npu/build/bin/moe-mapping-probe
python reports/2026-10-10/run-mapping.py
python scripts/analyze-moe-mapping.py reports/2026-10-10/mapping.jsonl \
  --train-trials 6 --tensor-sizes reports/2026-10-10/tensor-sizes.json \
  --output reports/2026-10-10/analysis.json
python tests/test_moe_mapping_analysis.py
```

The runner contains the tested local model path and replaces the run
artifacts when rerun. `tensor-sizes.json` contains the inspected tensor
metadata, allowing analysis without loading the GGUF. To inspect a
changed model, omit `--tensor-sizes` and put the pinned checkout's
`gguf-py` on `PYTHONPATH`; the model and its dependencies must be available.

Artifacts: `mapping.jsonl` (config and full per-trial counts/text),
`tensor-sizes.json` (GGUF tensor names, shapes, bytes), `run.json` (process
status and observed memory), and `analysis.json` (derived measurements).
The verbose runtime log is generated as `mapping.log` and is gitignored.

## Results and decision

All eight trials completed with exit code 0, no watchdog stop, and eight
distinct sampled continuations. Capture checks passed for all 40 layers:
9 prompt positions plus 23 decode evaluations per trial, with eight expert
selections per position per layer. Total wall time was 14.57
minutes. Four analysis tests passed, covering frozen held-out maps, tensor
byte accounting, unused MTP experts, invalid captures, and incomplete runs.
The C++ probe compiled with `-Wall -Wextra` without warnings.

| Trial | Mean union per layer | Layer minimum–maximum | New layer/expert pairs |
|---:|---:|---:|---:|
| 1 | 110.80 | 97–151 | 4432 |
| 2 | 116.75 | 100–165 | 238 |
| 3 | 123.15 | 104–178 | 256 |
| 4 | 126.20 | 104–184 | 122 |
| 5 | 126.78 | 105–184 | 23 |
| 6 | 131.20 | 107–191 | 177 |
| 7 | 134.95 | 109–200 | 150 |
| 8 | 145.62 | 119–215 | 427 |

The six-trial frozen map covered **100% of held-out prefill selections**
and **92.37% of held-out decode selections**
(13,597/14,720). The two held-out continuations
introduced 577 distinct layer/expert pairs absent from training.
The eighth trial alone added 427 pairs to the evolving union: the observed
set has **not converged**, despite the fifth trial's brief slowdown.
Coverage is selection-weighted, not a percentage of tokens or complete
forward passes that could run without fallback.

Weight payloads from the actual GGUF:

| Allocation | GiB |
|---|---:|
| All routed weights in the 40 active target layers | 8.71 |
| Six-trial training map, expert weights only | 4.47 |
| Eight-trial union, expert weights only | 4.96 |
| Every other GGUF tensor, including unused MTP block | 1.87 |
| Training map plus those other tensors | 6.34 |
| Final union plus those other tensors | 6.83 |
| Uniform 32-expert cache across target layers, expert payload only | 1.09 |
| Uniform 64-expert cache across target layers, expert payload only | 2.18 |

Minimum sampled system available memory was 3.61 GiB;
peak sampled process RSS was 5.52 GiB. Available memory changes
as Android reclaims file-backed pages, so this is a conservative sizing
reference rather than a proven allocation ceiling. The training-map expert
payload alone exceeds that minimum headroom, and other model/runtime
allocations must fit too. This experiment therefore does **not** satisfy
the conservative RAM gate for pinning the entire learned union.

Across trials, the process read **623.00 GiB** from storage, excluding
model loading. These are actual process I/O deltas. They do not show that
a cache would eliminate the same proportion of reads as its selection
coverage: weight sizes, page eviction, transfers, and synchronization
all matter. No cache or performance improvement was implemented or measured.

**Next implementation target:** a strictly bounded cache with full-model
fallback for every unseen or evicted expert, starting with a conservative
32-slot-per-layer budget (1.09 GiB raw routed-weight payload). Repeated-prompt
maps can inform retention or prefetching, but cannot safely remove experts.
The slot mechanism still needs router-ID remapping, synchronized weight
refills, and correctness comparison against the unchanged model before any
speed claim. Longer generations and different prompts remain untested here.

A subsequent implementation and full-model correctness gate are recorded
in the [bounded-cache follow-up](cache/README.md). The mapping measurements
above remain a separate experiment.
