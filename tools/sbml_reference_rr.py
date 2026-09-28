"""Reference trajectories from libroadrunner (run with the isolated temp site-packages first on sys.path)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.environ["TEMP"], "rr_verify"))
import numpy as np  # noqa: E402
import roadrunner  # noqa: E402

base = os.path.join(os.environ["TEMP"], "sbml_bench")
spec = json.load(open(os.path.join(base, "spec.json")))
out = {}
for mid, cfg in spec.items():
    try:
        rr = roadrunner.RoadRunner(os.path.join(base, f"{mid}.xml"))
        rr.integrator.relative_tolerance = 1e-9
        rr.integrator.absolute_tolerance = 1e-12
        ids = list(rr.model.getFloatingSpeciesIds())
        sel = ["time"] + [f"[{s}]" for s in ids]
        res = rr.simulate(0, cfg["t_end"], cfg["points"], selections=sel)
        arr = np.asarray(res)
        out[mid] = {"t": arr[:, 0].tolist(), "species": {s: arr[:, i + 1].tolist() for i, s in enumerate(ids)}}
    except Exception as exc:  # noqa: BLE001
        out[mid] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
json.dump(out, open(os.path.join(base, "reference.json"), "w"))
print("reference done:", {k: ("error" if "error" in v else len(v["species"])) for k, v in out.items()})
