"""Per-scene run-to-run spread, measured rather than remembered.

This repo has repeatedly cited a noise floor to decide whether a delta was
real. Those figures came from scattered ad-hoc repeats, and one of them --
"ficus has a 0.76 dB spread" -- turned out to describe a bug rather than the
scene, and was used to discount three conclusions.

Given two or more sweep JSONs over the same scenes, this prints the spread per
scene so the floor is a number with a provenance instead of a recollection.

    python bench/reproducibility.py a.json b.json [...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rows(fn: str) -> dict:
    d = json.loads(Path(fn).read_text())
    out = {}
    for r in d.get("rows", d.get("scenes", [])):
        if r.get("psnr") is not None:
            out[r["scene"]] = r
    return out, d


def main(files: list[str]) -> int:
    if len(files) < 2:
        raise SystemExit("need at least two sweep JSONs")
    sets = []
    for f in files:
        rs, d = rows(f)
        # Ask the structural check, not the marker. quarantine.py is a scan
        # run on demand, so a file written after the last scan carries no
        # marker and would otherwise report as fine while having no resolved
        # block at all -- which is exactly what happened to the first sweep.
        from bench.quarantine import has_provenance
        prov = "ok" if has_provenance(d) else "NO PROVENANCE"
        sets.append((Path(f).name, rs, prov))
        print(f"  {Path(f).name}  ({len(rs)} scenes, provenance {prov})")

    scenes = sorted(set().union(*[set(rs) for _, rs, _ in sets]))
    print(f"\n{'scene':<12} " + " ".join(f"{n[:14]:>14}" for n, _, _ in sets)
          + "   spread")
    spreads = []
    for sc in scenes:
        vals = [rs.get(sc, {}).get("psnr") for _, rs, _ in sets]
        have = [v for v in vals if v is not None]
        sp = (max(have) - min(have)) if len(have) > 1 else None
        if sp is not None:
            spreads.append((sc, sp))
        cells = " ".join(f"{v:14.2f}" if v is not None else f"{'--':>14}"
                         for v in vals)
        print(f"{sc:<12} {cells}   "
              + (f"{sp:.2f} dB" if sp is not None else "--"))

    if spreads:
        spreads.sort(key=lambda x: -x[1])
        worst, wv = spreads[0]
        mean = sum(v for _, v in spreads) / len(spreads)
        print(f"\n  mean spread {mean:.2f} dB, worst {worst} at {wv:.2f} dB "
              f"over {len(spreads)} scenes")
        print(f"  -> a per-scene delta below ~{wv:.2f} dB is not distinguishable "
              f"from noise on the worst scene;\n     below ~{mean:.2f} dB it is "
              f"not distinguishable on a typical one.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
