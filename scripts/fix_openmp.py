"""Deduplicate bundled libomp.dylib copies inside the venv.

Why this exists
---------------
Four wheels each vendor their own libomp.dylib (torch, open3d, sklearn,
pycolmap). Importing torch, then cv2/transformers, then pycolmap in one
process trips:

    OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib
    already initialized.

The widely-cited workaround is KMP_DUPLICATE_LIB_OK=TRUE. We do NOT use it:
it is documented by LLVM as unsafe and undocumented, and it can "silently
produce incorrect results" -- exactly the failure class this project exists
to catch. Instead we point every copy at a single real library.

Safety
------
We only symlink onto a donor that is verified to export every OMP symbol
that each consumer binary actually imports. If that check fails, we refuse
and leave the venv untouched. Idempotent; re-run after any `uv sync`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Consumer binaries whose undefined OMP symbols must remain satisfied.
CONSUMERS = [
    "open3d/cpu/pybind.cpython-312-darwin.so",
    "torch/lib/libtorch_cpu.dylib",
    "sklearn/_loss/_loss.cpython-312-darwin.so",
]


def _nm(path: Path, flag: str) -> set[str]:
    out = subprocess.run(
        ["nm", flag, str(path)], capture_output=True, text=True, check=False
    ).stdout
    syms = set()
    for line in out.splitlines():
        parts = line.split()
        if parts:
            syms.add(parts[-1])
    return syms


def exported(path: Path) -> set[str]:
    return _nm(path, "-gU")


def imported_omp(path: Path) -> set[str]:
    return {s for s in _nm(path, "-u") if "kmp" in s or s.startswith("_omp_")}


def find_site_packages(venv: Path) -> Path:
    hits = sorted(venv.glob("lib/python3.*/site-packages"))
    if not hits:
        sys.exit(f"no site-packages under {venv}")
    return hits[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venv", default=".venv", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sp = find_site_packages(args.venv)

    copies = [p for p in sp.rglob("libomp.dylib") if not p.is_symlink()]
    links = [p for p in sp.rglob("libomp.dylib") if p.is_symlink()]

    print(f"site-packages: {sp}")
    print(f"real libomp copies: {len(copies)}, existing symlinks: {len(links)}")
    for p in copies:
        print(f"  {p.relative_to(sp)}  {p.stat().st_size} bytes  {len(exported(p))} syms")

    if len(copies) <= 1:
        print("\nalready deduplicated -- nothing to do")
        return 0

    # Donor = the copy exporting the most symbols.
    donor = max(copies, key=lambda p: len(exported(p)))
    donor_syms = exported(donor)
    print(f"\ndonor: {donor.relative_to(sp)} ({len(donor_syms)} exported symbols)")

    # Verify every consumer's actually-imported OMP symbols survive the swap.
    ok = True
    for rel in CONSUMERS:
        c = sp / rel
        if not c.exists():
            print(f"  skip (absent): {rel}")
            continue
        need = imported_omp(c)
        missing = need - donor_syms
        status = "OK" if not missing else f"MISSING {len(missing)}"
        print(f"  {rel}: needs {len(need)} omp symbols -> {status}")
        if missing:
            ok = False
            for m in sorted(missing)[:10]:
                print(f"      {m}")

    if not ok:
        print("\nREFUSING: donor does not satisfy all consumers. venv untouched.")
        return 1

    print()
    for p in copies:
        if p == donor:
            continue
        if args.dry_run:
            print(f"  would link {p.relative_to(sp)} -> {donor.relative_to(sp)}")
            continue
        backup = p.with_suffix(".dylib.orig")
        if not backup.exists():
            p.rename(backup)
        elif p.exists():
            p.unlink()
        p.symlink_to(donor.resolve())
        print(f"  linked {p.relative_to(sp)} -> {donor.relative_to(sp)}")

    print("\ndone" + (" (dry run)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
