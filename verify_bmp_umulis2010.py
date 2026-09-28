"""Run and persist the Umulis et al. (2010) BMP model validation, and rebuild the browser cache.

    python verify_bmp_umulis2010.py

Exit code 0 only if every REQUIRED check passes (solver verification against the fully specified
2006 predecessor, the printed-Sog diagnostic, and the core 2010 claims). Every other check is
reported with its measured numbers, including the documented reproduction gaps.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from bmp_embryo import export_cache, validate


def main() -> int:
    report = validate(include_surface=True)
    out = Path("static/data/bmp/validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print("Umulis et al. 2010 BMP model - validation against the papers")
    print(f"{'ID':<5} {'RESULT':<7} {'KIND':<22} CLAIM")
    print("-" * 120)
    for row in report["checks"]:
        result = "PASS" if row["pass"] else ("FAIL" if row["pass"] is False else "-")
        print(f"{row['id']:<5} {result:<7} {row['kind']:<22} {row['claim']}")
        measured = {k: v for k, v in row["measured"].items() if k not in ("explanation", "note", "known_mismatch_15_min")}
        print("      " + json.dumps(measured, separators=(",", ":"), default=float)[:400])
    print("-" * 120)
    print(f"passed={report['passed']}/{report['total']} required={','.join(report['required_ids'])} "
          f"required_pass={report['required_pass']} wall_time_s={report['wall_time_s']:.1f}")
    print(f"validation_json={out} bytes={out.stat().st_size}")

    cache = export_cache()
    print(f"cache_json={cache['path']} bytes={cache['bytes']} surface_runs={','.join(cache['surface_runs'])} "
          f"cross_section_runs={len(cache['cross_section_runs'])} wall_time_s={cache['wall_time_s']:.1f}")
    if not report["required_pass"]:
        failed = [r["id"] for r in report["checks"] if r["id"] in report["required_ids"] and not r["pass"]]
        print("required_validation_failures=" + ",".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
