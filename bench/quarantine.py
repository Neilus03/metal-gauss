"""Mark result JSONs that cannot say what settings produced them.

A JSON without a `resolved` block records the harness's intent, not the run.
Two published tables were wrong for exactly that reason. These files stay in
git as history, but readme_tables.py refuses to render a table from one, so a
number with no provenance cannot reach the README by hand or by accident.

Re-run under bench/runner.py to clear the marker; nothing here edits data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "bench" / "results"
MARK = "provenance"


def has_provenance(d) -> bool:
    """Trainer results need a resolved config; kernel benchmarks need a commit.

    A result that reports PSNR came from a training run, and for those the only
    acceptable evidence is the trainer's own resolved configuration -- that is
    the thing whose absence let a harness publish 300k numbers labelled as
    something else. A kernel benchmark has no trainer config to resolve, so for
    those a git SHA and machine record is the bar.
    """
    if not isinstance(d, dict):
        return False
    rows = d.get("rows") or d.get("scenes")
    if isinstance(rows, list) and any(isinstance(r, dict) and
                                      r.get("psnr") is not None for r in rows):
        for r in rows:
            if r.get("psnr") is None:
                continue
            impl = r.get("impl")
            if impl in (None, "metal-gauss"):
                # our run: only the trainer's own resolved config will do
                if not isinstance(r.get("resolved"), dict):
                    return False
            else:
                # An external implementation has no resolved block and never
                # can -- it is not our trainer. What it must not be is a row
                # whose configuration is unrecorded, which is how a
                # schedule-scaled msplat curve got published as if it were
                # stock. Require something that names the variant.
                if not any(r.get(k) for k in
                           ("msplat_variant", "external_config", "cmd")):
                    return False
        return True
    if isinstance(d.get("resolved"), dict):
        return True
    if bool((d.get("env") or {}).get("git")):
        return True
    # A scorer artifact records a measurement of an artefact produced
    # elsewhere; its provenance is the run record it points at.
    return bool(d.get("ply") and d.get("scene") and d.get("views"))


def main(write: bool) -> int:
    marked = clean = 0
    for p in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"  UNPARSEABLE {p.name}: {e}")
            continue
        if has_provenance(d):
            if isinstance(d, dict) and d.get(MARK) == "incomplete":
                d.pop(MARK)
                if write:
                    p.write_text(json.dumps(d, indent=2))
            clean += 1
            print(f"  ok         {p.name}")
            continue
        marked += 1
        if isinstance(d, dict) and d.get(MARK) != "incomplete":
            d[MARK] = "incomplete"
            if write:
                p.write_text(json.dumps(d, indent=2))
        print(f"  QUARANTINE {p.name}")
    print(f"\n{clean} with provenance, {marked} quarantined"
          f"{'' if write else '  (dry run; pass --write)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main("--write" in sys.argv))
