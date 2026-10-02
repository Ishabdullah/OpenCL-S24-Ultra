"""Validate exact pinned source and patch without resetting a checkout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

def fail(message):
    sys.exit("Source validation failed: " + message)

def metadata(root):
    meta = json.loads((root / "upstream.json").read_text())
    patch = root / "patches/llama.cpp-qualcomm-sphal.patch"
    if hashlib.sha256(patch.read_bytes()).hexdigest() != meta["patch_sha256"]:
        fail("patch SHA-256 differs from upstream.json")
    return meta

def validate(root, source):
    meta = metadata(root)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != source.resolve():
        fail("source is not a separate checkout")
    if git("rev-parse", "HEAD") != meta["commit"]:
        fail("source HEAD differs from pin")
    if git("remote", "get-url", "origin") != meta["repository"]:
        fail("upstream origin differs from pin")
    changed = subprocess.check_output(["git", "-C", str(source), "ls-files", "-m", "-o", "--exclude-standard", "-z"]).decode().split("\0")
    unexpected = set(changed) - set(meta["patched_files_sha256"]) - {""}
    if unexpected:
        fail("unexpected local modifications: " + ", ".join(sorted(unexpected)))
    for name, expected in meta["patched_files_sha256"].items():
        p = source / name
        if not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            fail("patched file missing or changed: " + name)
    print("Pinned source and five patched files verified: " + meta["commit"])

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--metadata":
        meta = metadata(Path(sys.argv[2]))
        print(meta["repository"])
        print(meta["commit"])
    elif len(sys.argv) == 3:
        validate(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        sys.exit("Usage: validate-source.py PROJECT SOURCE | --metadata PROJECT")
