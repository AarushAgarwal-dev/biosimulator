"""Measured benchmark for nl_compiler (rules and Purdue gpt-oss:120b paths)."""
from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from nl_compiler import compile_text
from simulation_engine import ODEModel, derive_edges_from_odes, solve_pde


@dataclass
class Case:
    name: str
    text: str
    checks: List[Tuple[str, Callable[[Dict], bool]]]
    rule_designed: bool = True


def ir(bp):
    return bp.get("_ir", {}) if isinstance(bp, dict) else {}


def species_ids(bp):
    return {s.get("id") for s in ir(bp).get("species", [])}


def processes(bp, kind=None):
    ps = ir(bp).get("processes", [])
    return [p for p in ps if kind is None or p.get("kind") == kind]


def has_reg(bp, source, target, effect):
    return any(p.get("target") == target and any((r.get("species") or r.get("stimulus")) == source and r.get("effect") == effect
                                                   for r in p.get("regulators", [])) for p in processes(bp))


def has_conversion(bp, source, target):
    return any(p.get("kind") in ("conversion", "custom") and
               source in {x.get("species") for x in p.get("reactants", [])} and
               target in {x.get("species") for x in p.get("products", [])} for p in processes(bp))


def param_value(bp, prefix):
    values = [float(p["value"]) for p in ir(bp).get("parameters", []) if p.get("name", "").startswith(prefix)]
    return values[0] if values else None


def sim(bp, t=None):
    model = ODEModel(bp)
    return model.simulate(t or float(bp["simulation_config"]["t_max"]), 300)


def bounded(bp):
    if bp.get("validation_errors") or bp.get("type") == "PDE":
        return False
    result = sim(bp)
    values = np.concatenate([np.asarray(v) for v in result["species"].values()])
    return np.all(np.isfinite(values)) and np.min(values) >= -1e-6 and np.max(values) < 1e4


def process_param_value(bp, kind, field):
    ps = processes(bp, kind)
    if not ps:
        return None
    reference = ps[0].get(field)
    return next((float(p["value"]) for p in ir(bp).get("parameters", []) if p.get("name") == reference), None)


def edge(bp, source, target, kind):
    edges = derive_edges_from_odes(bp) or bp.get("edges", [])
    return any(e.get("source") == source and e.get("target") == target and e.get("type") == kind for e in edges)


def conserved(bp, combo):
    result = sim(bp, min(20, float(bp["simulation_config"]["t_max"])))
    total = sum(np.asarray(result["species"][name]) * coeff for name, coeff in combo)
    return float(np.ptp(total)) < 1e-5 * max(1.0, abs(float(total[0])))


def pde_pattern_forms(bp):
    if bp.get("type") != "PDE":
        return False
    spatial = bp["spatial"]; ids = list(spatial["reactions"])
    initial = {sid: {"type": "random_noise", "base_value": next(n["initial_value"] for n in bp["nodes"] if n["id"] == sid),
                     "noise_amplitude": 0.05} for sid in ids}
    state = np.random.get_state()
    try:
        np.random.seed(0)
        result = solve_pde(spatial, spatial["reactions"], initial,
                           float(bp["simulation_config"]["t_max"]), float(bp["simulation_config"]["dt"]), save_every=20)
    finally:
        np.random.set_state(state)
    return any(np.asarray(frames)[-1].std() > 3.0 * np.asarray(frames)[0].std()
               for frames in result["species"].values())



CASES = [
    Case("negative-autoregulation",
         "X starts at 1. X is produced at a constant rate of 2. X represses its own transcription. X is degraded with a half-life of 20 minutes. Simulate for 100 minutes.",
         [("self repression", lambda b: has_reg(b, "X", "X", "repress")),
          ("kdeg=ln2/20", lambda b: any(p.get("species") == "X" and abs(math.log(2)/float(p.get("half_life")) - math.log(2)/20) < 1e-12 for p in processes(b, "degradation"))),
          ("derived self inhibition", lambda b: edge(b, "X", "X", "inhibition"))]),
    Case("p53-mdm2",
         "Damage activates p53. p53 activates Mdm2 transcription. Mdm2 promotes degradation of p53. p53 is produced at rate 0.2. Mdm2 is degraded at rate 0.1. Simulate for 80 minutes.",
         [("two species", lambda b: {"p53", "Mdm2"} <= species_ids(b)),
          ("Mdm2 degrades p53", lambda b: any(p.get("species") == "p53" and any(r.get("species") == "Mdm2" and r.get("effect") == "activate" for r in p.get("regulators", [])) for p in processes(b, "degradation"))),
          ("bounded", bounded)]),
    Case("egfr-cascade",
         "EGF activates EGFR. EGFR activates RAS. RAS activates RAF. RAF activates MEK. MEK activates ERK. ERK inhibits EGFR. Simulate for 60 minutes.",
         [("six species", lambda b: len({"EGF","EGFR","RAS","RAF","MEK","ERK"} & species_ids(b)) == 6),
          ("feedback sign", lambda b: has_reg(b, "ERK", "EGFR", "repress")),
          ("ERK rises", lambda b: sim(b)["species"]["ERK"][-1] > sim(b)["species"]["ERK"][0])]),
    Case("glucose-chain",
         "Glucose is produced at a constant uptake rate of 2. Glucose is converted to Pyruvate at rate 0.5. Pyruvate is exported to Waste at rate 0.25. Simulate for 100 minutes.",
         [("two conversions", lambda b: len(processes(b, "conversion")) >= 2),
          ("all named", lambda b: {"Glucose","Pyruvate","Waste"} <= species_ids(b)),
          ("finite", bounded)]),
    Case("repressilator",
         "A represses B. B represses C. C represses A. A, B and C each start near 0.1. Simulate for 200 minutes.",
         [("three cyclic repressions", lambda b: all(has_reg(b, x, y, "repress") for x, y in (("A","B"),("B","C"),("C","A")))),
          ("asymmetric initials", lambda b: len({s["initial"] for s in ir(b).get("species", []) if s["id"] in {"A","B","C"}}) > 1),
          ("bounded", bounded)]),
    Case("toggle-switch",
         "A represses B with Hill coefficient 4. B represses A with Hill coefficient 4. A and B start at 0.1. Simulate for 100 minutes.",
         [("mutual repression", lambda b: has_reg(b,"A","B","repress") and has_reg(b,"B","A","repress")),
          ("Hill n=4", lambda b: sum(1 for p in ir(b).get("parameters", []) if p["name"].startswith("n_") and abs(float(p["value"])-4)<1e-9) >= 1),
          ("bounded", bounded)]),
    Case("ligand-binding",
         "L starts at 1. R starts at 1. L binds R reversibly to form C with kon 1 and koff 0.1. C is degraded at rate 0.05. Simulate for 20 minutes.",
         [("binding", lambda b: len(processes(b,"binding")) == 1),
          ("kon=1", lambda b: abs(process_param_value(b,"binding","kon")-1)<1e-12),
          ("koff=0.1", lambda b: abs(process_param_value(b,"binding","koff")-.1)<1e-12)]),
    Case("michaelis-menten",
         "E starts at 1. S starts at 10. E catalyzes the conversion of S to P with kcat 2 and Km 3. Simulate for 20 minutes.",
         [("enzyme conversion", lambda b: any(p.get("enzyme")=="E" and p.get("reactants",[{}])[0].get("species")=="S" for p in processes(b,"conversion"))),
          ("S+P conserved", lambda b: conserved(b, [("S",1),("P",1)])),
          ("E constant", lambda b: np.ptp(sim(b)["species"]["E"]) < 1e-9)]),
    Case("kinase-cycle",
         "Kinase catalyzes X to Xp. Phosphatase catalyzes Xp to X. X starts at 1 and Xp starts at 0. Simulate for 30 minutes.",
         [("forward", lambda b: has_conversion(b,"X","Xp")), ("reverse", lambda b: has_conversion(b,"Xp","X")),
          ("total X conserved", lambda b: conserved(b, [("X",1),("Xp",1)]))]),
    Case("central-dogma",
         "mRNA is produced at rate 1. mRNA is converted to Protein at rate 0.5. mRNA is degraded at rate 0.2. Protein is degraded at rate 0.1. Simulate for 50 minutes.",
         [("mRNA and Protein", lambda b: {"mRNA","Protein"} <= species_ids(b)),
          ("translation", lambda b: has_conversion(b,"mRNA","Protein")), ("bounded", bounded)]),
    Case("coherent-ffl",
         "A activates B. A activates C. B activates C. Simulate for 50 minutes.",
         [("three activations", lambda b: all(has_reg(b,x,y,"activate") for x,y in (("A","B"),("A","C"),("B","C")))),
          ("three species", lambda b: {"A","B","C"} <= species_ids(b)), ("bounded", bounded)]),
    Case("incoherent-ffl",
         "A activates B. A activates C. B represses C. Simulate for 50 minutes.",
         [("direct activation", lambda b: has_reg(b,"A","C","activate")),
          ("indirect repression", lambda b: has_reg(b,"B","C","repress")), ("finite", bounded)]),
    Case("lotka-volterra",
         "Prey X grows at rate 1.0 per hour. Predator Y consumes X at rate 0.1 with mass-action X times Y, and Y grows from consumption with yield 0.1. Y dies at rate 1.5. X starts at 10 and Y at 5. Simulate for 20 hours.",
         [("two species", lambda b: {"X","Y"} <= species_ids(b)), ("custom interaction", lambda b: any(p["kind"]=="custom" for p in processes(b))),
          ("finite", bounded)], False),
    Case("sir",
         "SIR model: infection converts S to I at rate beta*S*I with beta 0.3, and recovery converts I to R at rate gamma*I with gamma 0.1. S starts at 999, I at 1, R at 0. Simulate 160 days.",
         [("S I R", lambda b: {"S","I","R"} <= species_ids(b)), ("infection conversion", lambda b: has_conversion(b,"S","I")),
          ("population conserved", lambda b: conserved(b, [("S",1),("I",1),("R",1)]))], False),
    Case("logistic-growth",
         "Population N follows logistic growth dN/dt = r*N*(1-N/K), with r=0.5 and K=100. N starts at 1. Simulate for 30 days.",
         [("N species", lambda b: "N" in species_ids(b)), ("custom logistic rate", lambda b: any(p["kind"]=="custom" and "N" in p.get("rate","") for p in processes(b))),
          ("approaches K", lambda b: 90 <= sim(b)["species"]["N"][-1] <= 101)], False),
    Case("activator-inhibitor-pde",
         "Activator A diffuses at 0.05 and activates its own expression and inhibitor I. Inhibitor I diffuses at 1.0 and inhibits A. A and I start at 1. Simulate for 20 minutes.",
         [("PDE", lambda b: b.get("type")=="PDE"), ("D_I>D_A", lambda b: b["spatial"]["diffusion"]["I"] > b["spatial"]["diffusion"]["A"]),
          ("stable dt", lambda b: b["simulation_config"]["dt"]*max(b["spatial"]["diffusion"].values())*2 <= .5+1e-12),
          ("pattern amplifies", pde_pattern_forms)]),
    Case("brusselator-pde",
         "Reaction-diffusion Brusselator with U and V: dU/dt = 1-(4)*U+U**2*V, dV/dt = 3*U-U**2*V. U diffuses at 0.01 and V at 0.1. U starts at 1 and V at 3. Simulate for 10 seconds.",
         [("PDE", lambda b: b.get("type")=="PDE"), ("numeric reactions", lambda b: all("k_" not in x for x in b["spatial"]["reactions"].values())),
          ("stable", lambda b: b["_verification"]["pde_stability"]["stable"])], False),
    Case("calcium-transport",
         "CalciumStore is converted to CytosolicCalcium at rate 0.2. CytosolicCalcium is converted to CalciumStore at rate 0.1. CalciumStore starts at 10 and CytosolicCalcium starts at 0. Simulate for 50 seconds.",
         [("bidirectional transport", lambda b: has_conversion(b,"CalciumStore","CytosolicCalcium") and has_conversion(b,"CytosolicCalcium","CalciumStore")),
          ("calcium conserved", lambda b: conserved(b, [("CalciumStore",1),("CytosolicCalcium",1)])),
          ("nonnegative", lambda b: min(np.concatenate([np.asarray(v) for v in sim(b)["species"].values()])) >= -1e-8)]),
    Case("all-explicit-numbers",
         "A starts at 2 and is produced at rate 1.5. A is degraded at rate 0.3. Simulate for 20 minutes.",
         [("initial text", lambda b: next(s for s in ir(b)["species"] if s["id"]=="A")["initial_source"]=="text"),
          ("production text", lambda b: any(p["source"]=="text" and abs(float(p["value"])-1.5)<1e-12 for p in ir(b)["parameters"])),
          ("degradation text", lambda b: any(p["source"]=="text" and abs(float(p["value"])-.3)<1e-12 for p in ir(b)["parameters"]))]),
    Case("negation",
         "A does not activate B. A is degraded at rate 0.1. Simulate for 10 minutes.",
         [("no A-to-B process", lambda b: not has_reg(b,"A","B","activate")), ("A degradation", lambda b: any(p.get("species")=="A" for p in processes(b,"degradation"))),
          ("no B species", lambda b: "B" not in species_ids(b))]),
    Case("non-biological",
         "The weather is pleasant and the conference begins tomorrow.",
         [("clean error", lambda b: bool(b.get("validation_errors"))), ("no fabricated species", lambda b: not ir(b).get("species")),
          ("no crash marker", lambda b: not any("Traceback" in e for e in b.get("validation_errors",[])))], False),
]


def live_config():
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=False)
    except ImportError:
        return None
    key = os.getenv("PURDUE_GENAI_API_KEY")
    if not key:
        return None
    # Feature-detect the concurrently-added Purdue engine without ever rendering the key.
    try:
        import llm_provider
        llm_provider.build_client({"engine": "purdue", "model": "gpt-oss:120b"})
        return {"engine": "purdue", "model": "gpt-oss:120b"}
    except Exception:
        return {"engine": "remote", "base_url": "https://genai.rcac.purdue.edu/api",
                "model": "gpt-oss:120b", "api_key": key}


def run_path(label: str, config, selected=CASES):
    rows = []
    for case in selected:
        start = time.perf_counter()
        bp = compile_text(case.text, config)
        latency = time.perf_counter() - start
        assertions = []
        for name, check in case.checks:
            try:
                ok = bool(check(bp))
            except Exception as exc:
                ok = False
                name += f" [{type(exc).__name__}]"
            assertions.append((name, ok))
        passed = all(ok for _, ok in assertions)
        rows.append({"case": case, "path": label, "pass": passed, "latency": latency,
                     "assertions": assertions, "errors": bp.get("validation_errors", [])})
        marks = ", ".join(f"{'PASS' if ok else 'FAIL'}:{name}" for name, ok in assertions)
        print(f"{case.name:27} {label:10} {'PASS' if passed else 'FAIL':4} {latency:8.3f}s  {marks}")
    return rows


def summarize(rule_rows, live_rows):
    designed = [r for r in rule_rows if r["case"].rule_designed]
    rule_rate = sum(r["pass"] for r in designed) / max(1, len(designed))
    print(f"\nRule designed cases: {sum(r['pass'] for r in designed)}/{len(designed)} = {rule_rate:.1%} (target >=80%)")
    live_rate = None
    if live_rows:
        live_rate = sum(r["pass"] for r in live_rows) / len(live_rows)
        print(f"LLM overall cases:    {sum(r['pass'] for r in live_rows)}/{len(live_rows)} = {live_rate:.1%} (target >=90%)")
    else:
        print("LLM overall cases:    SKIPPED (PURDUE_GENAI_API_KEY unavailable)")
    failures = [r for r in rule_rows + live_rows if not r["pass"]]
    if failures:
        print("\nFailures:")
        for row in failures:
            failed = [name for name, ok in row["assertions"] if not ok]
            detail = "; ".join(row["errors"][:2]) if row["errors"] else "assertion mismatch"
            print(f"- {row['path']}/{row['case'].name}: {', '.join(failed)} -- {detail}")
    return rule_rate, live_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules-only", action="store_true")
    parser.add_argument("--case", action="append", help="run only named case (repeatable)")
    args = parser.parse_args()
    selected = [c for c in CASES if not args.case or c.name in set(args.case)]
    print(f"BioSimulateAI nl_compiler benchmark: {len(selected)} cases; assertions are shown individually.\n")
    rule_rows = run_path("rules", None, selected)
    cfg = None if args.rules_only else live_config()
    live_rows = run_path("llm", cfg, selected) if cfg else []
    rule_rate, live_rate = summarize(rule_rows, live_rows)
    target_rule = rule_rate >= .80
    target_live = live_rate is None or live_rate >= .90
    raise SystemExit(0 if target_rule and target_live else 1)


if __name__ == "__main__":
    main()
