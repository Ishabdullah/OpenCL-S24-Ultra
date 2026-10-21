# Cache-80 HVX/MTP factorial

Previous detached attempt remains in qualification/ (plan only). The short detached
preflight smoke is qualification-detached-smoke-01/; it performs no cooling or
inference. qualification-retry-01/ is reserved for the real retry and must not
be launched without separate authorization. Launch it from a supervised terminal
with the command: python launch.py qualification-retry-01. The launcher uses a new session, redirects
stdin/stdout/stderr, writes a PID and log, and returns immediately. Inspect the
PID, lifecycle preflight, telemetry, and eventual summary; a PID alone is not
proof of success. Do not run from a timed shell command using nohup alone: that
keeps the driver in the timed command's process group.

Offline preview: `python run.py --stage qualification` or `python run.py --stage screen`. Tests: `python -m unittest test_run.py`. No inference is launched without `--run`. Run qualification first; inspect its evidence before separately authorizing screen. Each stage writes into its own new directory and refuses to overwrite it. Do not run both stages concurrently.

Qualification: H4/T4 and H1/T6, off then on, 16 tokens. Screen: thread-major T4,T5,T6; H4,H3,H2,H1 within each thread; off then on, 64 tokens (24 cells). Cooling before each cell lasts at least 180 seconds and requires B<=40, CPU<=55, HTP<=50; after 300 seconds without reaching targets the run aborts. Charging and >=40% battery are required. 5 GiB MemAvailable is preferred; below 1.5 GiB is refused. Per-cell raw stdout/stderr, lifecycle, telemetry, results and summary remain on disk. No automatic retry on failed or interrupted stages; use a fresh evidence directory after review.
