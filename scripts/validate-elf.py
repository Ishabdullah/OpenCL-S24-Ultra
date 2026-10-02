"""Verify pinned build options and project ELF dependencies without loading vendor code."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

def validate(root):
    build = root / ".work/build"
    source = root / ".work/llama.cpp"
    bins = build / "bin"
    cache = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if "=" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            cache[key.split(":", 1)[0]] = value
    expected = {
        "CMAKE_BUILD_TYPE": "Release", "GGML_OPENCL": "ON",
        "GGML_OPENCL_USE_SPHAL": "ON", "GGML_OPENCL_TARGET_VERSION": "300",
        "GGML_OPENCL_USE_ADRENO_KERNELS": "OFF", "GGML_OPENCL_USE_ADRENO_BIN_KERNELS": "OFF",
        "GGML_OPENCL_EMBED_KERNELS": "ON", "GGML_OPENCL_PROFILING": "OFF",
        "GGML_BACKEND_DL": "OFF", "CMAKE_HOME_DIRECTORY": str(source),
    }
    for key, value in expected.items():
        if cache.get(key) != value:
            raise ValueError(f"{key}={cache.get(key)!r}, expected {value!r}. Run ./install.sh --rebuild.")
    reader = shutil.which("llvm-readelf")
    if not reader:
        raise ValueError("llvm-readelf missing; reinstall Termux clang (depends on llvm)")
    backend = bins / "libggml-opencl.so.0"
    if not backend.is_file():
        raise ValueError("OpenCL backend missing")
    records = {}
    for p in sorted(bins.iterdir()):
        if p.is_symlink() or not p.is_file():
            continue
        with p.open("rb") as f:
            if f.read(4) != b"\x7fELF":
                continue
        dynamic = subprocess.check_output([reader, "--dynamic", str(p)], text=True)
        needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
        if any(name.startswith("libOpenCL") for name in needed):
            raise ValueError(f"{p.name} directly needs an OpenCL library: {needed}")
        for name in needed:
            if name.startswith(("libllama", "libggml")):
                target = bins / name
                if not target.is_file() or target.resolve().parent != bins.resolve():
                    raise ValueError(f"{p.name}: {name} absent/outside build; installed llama could be selected")
        records[p.name] = needed
    versions = subprocess.check_output([reader, "--version-info", str(backend)], text=True)
    symbols = subprocess.check_output([reader, "--dyn-syms", "--wide", str(backend)], text=True)
    if re.search(r"OPENCL_\w+", versions):
        raise ValueError("OpenCL ELF version requirement found")
    if re.search(r"\bUND\s+cl[A-Z]\w+", symbols):
        raise ValueError("unresolved OpenCL API instead of internal dispatch")
    result = {"configuration_verified": True, "no_OpenCL_DT_NEEDED": True,
              "no_OPENCL_symbol_versions": True, "no_undefined_OpenCL_APIs": True,
              "all_project_dependencies_present": True, "needed": records}
    logs = root / ".work/logs"
    logs.mkdir(exist_ok=True)
    (logs / "elf.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"ELF verified: {len(records)} project binaries/libraries; no OpenCL DT_NEEDED, versions or unresolved APIs.")

if __name__ == "__main__":
    try:
        validate(Path(sys.argv[1]).resolve())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        sys.exit("ELF validation failed: " + str(error))
