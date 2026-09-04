"""Build the portable Metal IR library used by the runtime loader.

This is intentionally a small, explicit build step rather than an import-time
side effect. The repository still works from a source checkout without Xcode;
when the Metal command-line tools are available, this script emits one
``metal_gauss.metallib`` plus a source digest sidecar. The digest lets the
runtime reject a stale checked-in or installed library and fall back to source.

On a normal macOS developer installation the tools are available through
``xcrun``. Command Line Tools without the full Xcode Metal toolchain will
report a useful error and can continue using runtime compilation.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
    "rasterize.metal",
    "preprocess.metal",
    "adam.metal",
    "ssim.metal",
    "binning.metal",
)


def source_bytes() -> bytes:
    return b"\n".join((ROOT / "metal_gauss" / "csrc" / name).read_bytes()
                         for name in SOURCE_NAMES) + b"\n"


def run_tool(tool: str, args: list[str]) -> None:
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        raise SystemExit("xcrun is required to build a Metal library")
    try:
        subprocess.run([xcrun, "-sdk", "macosx", tool, *args], check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"xcrun could not find {tool!r}; install Xcode's Metal tools or "
            "use the runtime-source fallback"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{tool} failed with exit code {exc.returncode}") from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "metal_gauss" / "csrc" / "metal_gauss.metallib")
    args = ap.parse_args()
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="metal-gauss-metallib-") as tmp:
        tmp_path = Path(tmp)
        air = []
        for name in SOURCE_NAMES:
            src = ROOT / "metal_gauss" / "csrc" / name
            dst = tmp_path / f"{src.stem}.air"
            run_tool("metal", ["-c", str(src), "-o", str(dst)])
            air.append(str(dst))
        run_tool("metallib", [*air, "-o", str(out)])

    digest = hashlib.sha256(source_bytes()).hexdigest()
    out.with_suffix(out.suffix + ".sha256").write_text(
        digest + "\n", encoding="ascii"
    )
    print(f"wrote {out}")
    print(f"source sha256 {digest}")


if __name__ == "__main__":
    main()
