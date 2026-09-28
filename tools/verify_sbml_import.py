"""Verify sbml_import against libroadrunner on curated BioModels entries.

    python tools/verify_sbml_import.py            # needs network + libroadrunner in %TEMP%\\rr_verify

For each model the SBML is downloaded from BioModels, simulated by libroadrunner (the reference
SBML simulator) in a separate process, and by BioSimulateAI's importer + ODEModel. The maximum
relative error over all floating species and time points (relative to each species' range) is
reported; a model passes below 1 %.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import db_interface  # noqa: E402
import sbml_import  # noqa: E402
from simulation_engine import ODEModel  # noqa: E402

MODELS = {
    "BIOMD0000000010": {"t_end": 6000.0, "points": 601, "about": "Kholodenko 2000 MAPK oscillations"},
    "BIOMD0000000012": {"t_end": 1000.0, "points": 501, "about": "Elowitz & Leibler 2000 repressilator"},
    "BIOMD0000000005": {"t_end": 100.0, "points": 501, "about": "Tyson 1991 cell cycle"},
    "BIOMD0000000009": {"t_end": 100.0, "points": 501, "about": "Huang & Ferrell 1996 MAPK ultrasensitivity"},
    "BIOMD0000000021": {"t_end": 200.0, "points": 501, "about": "Leloup & Goldbeter 1998 Drosophila circadian"},
    "BIOMD0000000028": {"t_end": 1000.0, "points": 501, "about": "Markevich 2004 MAPK double phosphorylation"},
    "BIOMD0000000035": {"t_end": 200.0, "points": 501, "about": "Vilar 2002 circadian oscillator"},
    "BIOMD0000000043": {"t_end": 20.0, "points": 501, "about": "Borghans 1997 calcium oscillations"},
}


def main() -> int:
    base = Path(os.environ["TEMP"]) / "sbml_bench"
    base.mkdir(exist_ok=True)
    spec = {}
    for mid, cfg in MODELS.items():
        path = base / f"{mid}.xml"
        if not path.exists():
            text = db_interface.fetch_biomodel_sbml(mid)
            if not text:
                print(f"{mid}: download failed")
                continue
            path.write_text(text, encoding="utf-8")
        spec[mid] = cfg
    (base / "spec.json").write_text(json.dumps(spec))
    ref_script = ROOT / "tools" / "sbml_reference_rr.py"
    subprocess.run([sys.executable, str(ref_script)], check=True)
    reference = json.loads((base / "reference.json").read_text())

    passed = 0
    for mid, cfg in spec.items():
        ref = reference.get(mid, {})
        if "error" in ref:
            print(f"{mid:17s} SKIP  reference failed: {ref['error'][:120]}")
            continue
        try:
            bp = sbml_import.sbml_to_blueprint((base / f"{mid}.xml").read_text(encoding="utf-8"))
            ours = ODEModel(bp).simulate(cfg["t_end"], num_points=cfg["points"])
        except Exception as exc:  # noqa: BLE001
            print(f"{mid:17s} FAIL  import/simulate raised {type(exc).__name__}: {str(exc)[:160]}")
            continue
        sbml_to_ours = {n.get("sbml_id"): n["id"] for n in bp["nodes"]}
        worst, worst_sp = 0.0, None
        compared = 0
        for sid, series in ref["species"].items():
            ours_id = sbml_to_ours.get(sid)
            if ours_id is None or ours_id not in ours["species"]:
                continue
            a = np.asarray(series)
            b = np.asarray(ours["species"][ours_id])
            scale = max(float(np.ptp(a)), float(np.max(np.abs(a))) * 1e-3, 1e-12)
            err = float(np.max(np.abs(a - b)) / scale)
            compared += 1
            if err > worst:
                worst, worst_sp = err, sid
        ok = compared > 0 and worst < 0.01
        passed += ok
        print(f"{mid:17s} {'PASS' if ok else 'FAIL'}  species compared={compared:2d}  max rel. error={worst:.2e}"
              f" ({worst_sp})  {cfg['about']}")
    print(f"\n{passed}/{len(spec)} models match libroadrunner within 1%")
    return 0 if passed == len(spec) else 1


if __name__ == "__main__":
    raise SystemExit(main())
