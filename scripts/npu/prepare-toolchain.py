#!/data/data/com.termux/files/usr/bin/python
"""Pinned SDK/tool runtime in project-private storage; no system installation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(record, path):
    path = Path(path)
    if path.is_symlink():
        raise RuntimeError('Download destination must not be a symlink: ' + str(path))
    if path.exists():
        if path.stat().st_size == record['size'] and digest(path) == record['sha256']:
            print('Verified cached download: ' + path.name, flush=True)
            return path
        raise RuntimeError('Existing download has the wrong checksum; move it aside: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    if part.is_symlink():
        raise RuntimeError('Partial download must not be a symlink: ' + str(part))
    urls = record.get('urls', [record.get('url')])
    for url in urls:
        print('Download: ' + url, flush=True)
        result = subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                                 '--connect-timeout', '30', '--max-time', '3600',
                                 '--proto', '=https', '--proto-redir', '=https',
                                 '--output', str(part), url])
        if result.returncode == 0:
            if part.stat().st_size != record['size'] or digest(part) != record['sha256']:
                raise RuntimeError('Downloaded bytes differ from the pinned size/SHA-256: ' + str(part))
            part.replace(path)
            return path
    raise RuntimeError('Pinned download unavailable; URLs and hashes are in npu-upstream.json: ' + path.name)


def prepare(root, archive):
    meta = json.loads((root / 'npu-upstream.json').read_text())
    work = root / '.work-npu'
    tools_root = work / 'toolchain'
    cache = work / 'downloads'
    sdk = tools_root / 'sdk' / meta['sdk']['version']
    x86 = tools_root / 'x86-root'
    emulation = tools_root / 'emulation'
    for p in [tools_root, cache, sdk.parent, x86, emulation]:
        if p.is_symlink():
            raise RuntimeError('Managed toolchain directory is a symlink: ' + str(p))
        p.mkdir(parents=True, exist_ok=True)
    archive_path = Path(archive).expanduser().resolve() if archive else cache / 'hexagon-sdk-v6.6.0.0-amd64-lnx.tar.xz'
    if archive:
        if archive_path.stat().st_size != meta['sdk']['size'] or digest(archive_path) != meta['sdk']['sha256']:
            raise RuntimeError('--sdk-archive differs from the pinned archive checksum')
    else:
        archive_path = fetch(meta['sdk'], archive_path)
    stamp = tools_root / 'sdk-extracted.json'
    if sdk.exists() and not stamp.exists():
        raise RuntimeError('Incomplete/unmarked SDK extraction. Move .work-npu/toolchain/sdk aside and rerun.')
    if not sdk.exists():
        if shutil.disk_usage(work).free < 5 * 1024**3:
            raise RuntimeError('At least 5 GiB free is required for SDK extraction and the source build.')
        print('Extract verified SDK archive (about 3.2 GiB unpacked)', flush=True)
        # The verified archive has one SDK directory. Data filtering rejects
        # external paths and unsafe links while retaining in-tree SDK symlinks.
        with tarfile.open(archive_path, 'r:xz') as tar:
            tar.extractall(sdk.parent, filter='data')
        if not (sdk / 'build/cmake/hexagon_fun.cmake').is_file():
            raise RuntimeError('SDK archive layout differs from the tested version')
        stamp.write_text(json.dumps({'sha256': meta['sdk']['sha256'], 'version': meta['sdk']['version']}) + '\n')
    elif json.loads(stamp.read_text()).get('sha256') != meta['sdk']['sha256']:
        raise RuntimeError('SDK extraction stamp differs from the pin')
    runtime_stamp = tools_root / 'x86-runtime.json'
    expected = {p['name']: p['sha256'] for p in meta['x86_runtime']}
    for package in meta['x86_runtime']:
        deb = fetch(package, cache / package['filename'])
        if not runtime_stamp.exists() or json.loads(runtime_stamp.read_text()) != expected:
            subprocess.run(['dpkg-deb', '--extract', str(deb), str(x86)], check=True)
    runtime_stamp.write_text(json.dumps(expected, indent=2) + '\n')
    # Debian's merged-/usr packages do not include these filesystem aliases.
    # They are required by QEMU's ELF interpreter/library lookup inside this root.
    for name, target_path in [('lib', 'usr/lib'), ('lib64', 'usr/lib64')]:
        link = x86 / name
        if link.is_symlink():
            if os.readlink(link) != target_path:
                raise RuntimeError('Unexpected x86 runtime link: ' + str(link))
        elif link.exists():
            raise RuntimeError('Expected merged-/usr runtime alias: ' + str(link))
        else:
            link.symlink_to(target_path)
    bin_dir = emulation / 'bin'
    bin_dir.mkdir(exist_ok=True)
    for name in ['qaic', 'hexagon-clang', 'hexagon-clang++', 'hexagon-ar', 'hexagon-link', 'ld.qcld']:
        dest = bin_dir / name
        if dest.is_symlink():
            raise RuntimeError('Compiler wrapper must be a file, not a symlink: ' + str(dest))
        shutil.copyfile(root / 'scripts/npu/sdk-tool.py', dest)
        dest.chmod(0o700)
    target = sdk / 'tools/HEXAGON_Tools' / meta['sdk']['tools_version'] / 'Tools/target/hexagon'
    (emulation / 'target').mkdir(exist_ok=True)
    for link in [bin_dir / 'hexagon', emulation / 'target/hexagon']:
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError('SDK target symlink points outside this installation: ' + str(link))
        elif link.exists():
            raise RuntimeError('Expected SDK target symlink, found directory/file: ' + str(link))
        else:
            link.symlink_to(target)
    env = os.environ.copy()
    env.pop('LD_LIBRARY_PATH', None)
    env.pop('LD_PRELOAD', None)
    subprocess.run([str(bin_dir / 'hexagon-clang'), '--version'], env=env, check=True)
    print('SDK/compiler runtime prepared; emulation is for build tools only.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('project', type=Path)
    parser.add_argument('--sdk-archive')
    args = parser.parse_args()
    try:
        prepare(args.project.resolve(), args.sdk_archive)
    except (OSError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError) as error:
        raise SystemExit('NPU toolchain preparation failed: ' + str(error))
