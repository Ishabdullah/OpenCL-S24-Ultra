# Safe Factorial Harness (Cache-80 HVX/MTP)

This module provides a safe, telemetry-instrumented test harness for running multi-cell factorials across HVX thread and MTP configurations under bounded expert caching (80 slots / 3200 MiB).

## Root Cause of Prior Failure (retry-01 collapse)

Under `LLAMA_LOAD_MODE_AUTO`, llama.cpp inspects every registered device. `ggml-hexagon.cpp` hardcodes `mmap_support = false`. When any device reports false, the loader eagerly disables mmap for the entire model and attempts to read the multi-gigabyte GGUF file directly into process RSS. On this 10 GiB device, this caused available RAM to plummet from 4.46 GiB to 1.85 GiB in ~8.35 seconds, risking an OOM death spiral.

## Concrete Safety Fixes

All child commands are strictly pinned with the following parameters:
- `-lm mmap`: Overrides the Hexagon mmap disable veto, ensuring zero eager full-file RSS loading.
- `-dev HTP0`: Offloads compute to the Hexagon v75 NPU.
- `-ngl 99`: Offloads all 40 transformer blocks.
- `-c 256`: Context window fixed at 256.
- `-b 128 -ub 128`: Batch and microbatch sizes pinned to 128.
- `-np 1`: Single sequence generation.
- `-fa off`: Flash attention disabled to prevent MTP race conditions.

## Safety & Telemetry Architecture

1. **Strict Admission Gate**:
   - Device must be charging and battery must be >= 40%.
   - Minimum MemAvailable >= 5 GiB eligibility floor, factored with free swap backstopping.
   - All thermal sensors (`b`, `c`, `htp`) must be online and responsive.
2. **Early-Abort Trend Projections**:
   - Least-squares regression slopes over recent memory and swap telemetry points.
   - If projected memory crosses below the 2 GiB retained headroom floor or swap crosses below the 512 MiB floor, the cell is terminated *immediately* before the floor is breached.
3. **Watchdog & Process Management**:
   - Non-blocking background thread refreshes battery status asynchronously; watchdog ticks never hang on external subprocess calls.
   - Independent process groups (`start_new_session=True`) allow clean group teardown on aborts.
   - Single-run locking (`RunLock` via `fcntl.flock`) prevents concurrent driver invocations.
4. **Verification**:
   - Comprehensive test suite in `test_safe_run.py` covers all safety mechanisms, trend projections, and replays the historical collapse trace without launching live inference.

## Usage

```sh
# Run non-inference test suite (19 unit tests)
python -m unittest discover -s reports/2026-10-13/safe-factorial-harness -p "test_*.py"

# Preview plan and pin validation (dry-run, no inference launched)
python reports/2026-10-13/safe-factorial-harness/safe_run.py --stage qualification --evidence dry-run-check

# Run non-inference prerequisite gate
python reports/2026-10-13/safe-factorial-harness/safe_run.py --screen-prereq
```
