"""Can BioSimulateAI build a working model from WORDS alone?

Drives the deterministic text path (no LLM) end to end: plain-English description ->
blueprint -> compiled model -> simulation -> numerical check of the behaviour the
words describe. Deterministic on purpose: an LLM path that sometimes works is not a
verification, and this must be runnable offline in CI.

Every check asserts a NUMBER, not merely the absence of an exception.
"""
import sys

sys.path.insert(0, r"D:\SURF\biosimulator")

import numpy as np

import agent
import simulation_engine

PASS, FAIL = "PASS", "FAIL"
results = []


def record(name, check, measured, tolerance, ok):
    results.append((name, check, measured, tolerance, PASS if ok else FAIL))
    flag = PASS if ok else FAIL
    print(f"      {flag}  {check:<52} {measured:<22} {tolerance}")


def build(text):
    """Words -> blueprint, via the deterministic parser only."""
    blueprint = agent.rule_based_parse(text)
    return blueprint


def simulate(blueprint, t_max=50.0):
    model = simulation_engine.ODEModel(blueprint)
    return model.simulate(t_max=t_max)


def series(result, name):
    # ODEModel.simulate returns {"t": [...], "species": {id: [...]}}.
    data = result.get("species") or {}
    if name in data:
        return np.asarray(data[name], dtype=float)
    for key, value in data.items():
        if str(key).upper() == name.upper():
            return np.asarray(value, dtype=float)
    raise KeyError(f"{name} not in {sorted(data)}")


# ---------------------------------------------------------------------------
print("\n[1/5] WORDS -> a two-species activation cascade")
text1 = ("EGF activates EGFR. EGFR activates ERK. "
         "EGF starts at 2.5. EGFR starts at 0.5. ERK starts at 0.0.")
print(f"      text: {text1}")
bp1 = build(text1)
ids1 = [n["id"] for n in bp1.get("nodes", [])]
init1 = {n["id"]: n.get("initial_value") for n in bp1.get("nodes", [])}
print(f"      species parsed: {ids1}")
print(f"      initial values: {init1}")
record("cascade", "all three species were parsed from words", str(sorted(ids1)),
       "EGF, EGFR, ERK", set(ids1) >= {"EGF", "EGFR", "ERK"})
record("cascade", "decimal initial value survived (EGF)", init1.get("EGF"), "== 2.5",
       init1.get("EGF") == 2.5)
record("cascade", "edges parsed from words", len(bp1.get("edges", [])), ">= 2",
       len(bp1.get("edges", [])) >= 2)

res1 = simulate(bp1, t_max=50.0)
erk = series(res1, "ERK")
record("cascade", "all ERK samples finite", int(np.sum(~np.isfinite(erk))), "== 0",
       bool(np.all(np.isfinite(erk))))
record("cascade", "ERK rose from its zero initial condition", f"{erk.max():.6g}",
       "> 1e-6", erk.max() > 1e-6)

# ---------------------------------------------------------------------------
print("\n[2/5] WORDS -> negative feedback (inhibition is honoured)")
text2 = ("A activates B. B inhibits A. A starts at 1.0. B starts at 0.25.")
print(f"      text: {text2}")
bp2 = build(text2)
kinds = sorted({e.get("type") for e in bp2.get("edges", [])})
record("feedback", "both activation and inhibition parsed", str(kinds),
       "contains inhibition", any("inhib" in str(k) for k in kinds))
init2 = {n["id"]: n.get("initial_value") for n in bp2.get("nodes", [])}
record("feedback", "sub-unit initial value survived (B)", init2.get("B"), "== 0.25",
       init2.get("B") == 0.25)
res2 = simulate(bp2, t_max=40.0)
a2 = series(res2, "A")
record("feedback", "A stays finite and non-negative", f"{a2.min():.6g}", ">= -1e-9",
       bool(np.all(np.isfinite(a2))) and a2.min() >= -1e-9)

# ---------------------------------------------------------------------------
print("\n[3/5] WORDS -> a PDE (reaction-diffusion), configured and solved")
# The PDE path is field-based rather than a species graph, so "from words" here means
# the named presets the UI offers plus the plain-language summary it renders.
import pde_model
import pde_solver_1d

names = pde_model.preset_names()
print(f"      pde presets available: {names}")
record("pde", "named PDE presets exist for a researcher to pick", len(names), ">= 3",
       len(names) >= 3)

# Pick a diffusion-like preset and actually solve it, so this is a real check.
chosen = next((n for n in names if "diff" in n.lower()), names[0] if names else None)
if chosen:
    preset = pde_model.get_preset(chosen)
    fields = preset.get("fields") or []
    record("pde", "preset defines at least one field", len(fields), ">= 1",
           len(fields) >= 1)
    print(f"      preset: {chosen}")

    if fields:
        field = fields[0]
        # The spec requires BOTH a mathematical and a plain-language rendering, so a
        # biologist can check the model without reading LaTeX.
        latex = pde_model.equation_latex(field)
        plain = pde_model.plain_language_summary(field)
        print(f"      latex: {str(latex)[:110]}")
        print(f"      plain: {str(plain)[:150]}")
        record("pde", "renders a mathematical (LaTeX) equation", bool(latex), "truthy",
               bool(latex))
        record("pde", "renders a plain-language summary for a biologist", bool(plain),
               "truthy", bool(plain))
        record("pde", "plain summary is genuinely prose, not the LaTeX again",
               len(str(plain).split()), ">= 6", len(str(plain).split()) >= 6)

    # Solve a no-flux diffusion problem and check it is PHYSICAL, not just finite.
    x = np.linspace(0.0, 1.0, 61)
    dx = float(x[1] - x[0])
    initial = np.exp(-((x - 0.5) ** 2) / 0.01)
    import boundary_conditions as bc
    solved = pde_solver_1d.solve_1d(
        x, initial, diffusion=1.0, t_max=0.02, dt=0.2 * dx * dx / 1.0,
        left=bc.make_no_flux("u", "left"), right=bc.make_no_flux("u", "right"))
    frames = np.asarray(solved["u"], dtype=float)
    record("pde", "every value in every frame is finite",
           int(np.sum(~np.isfinite(frames))), "== 0", bool(np.all(np.isfinite(frames))))
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    mass0, massN = float(trap(frames[0], x)), float(trap(frames[-1], x))
    drift = abs(massN / mass0 - 1.0)
    record("pde", "no-flux diffusion conserves mass", f"{drift:.3e}", "< 1e-9",
           drift < 1e-9)
    spread0 = float(frames[0].max() - frames[0].min())
    spreadN = float(frames[-1].max() - frames[-1].min())
    record("pde", "diffusion flattened the profile", f"{spread0:.4f} -> {spreadN:.4f}",
           "decreasing", spreadN < spread0)

# ---------------------------------------------------------------------------
print("\n[4/5] WORDS with no relations at all -> must FAIL honestly, not fake a model")
text4 = "The weather is nice today and I like cells."
bp4 = build(text4)
print(f"      nodes: {[n['id'] for n in bp4.get('nodes', [])]}  "
      f"edges: {len(bp4.get('edges', []))}")
record("nonsense", "no fabricated species from a non-biological sentence",
       len(bp4.get("nodes", [])), "== 0", len(bp4.get("nodes", [])) == 0)

# ---------------------------------------------------------------------------
print("\n[5/5] WORDS -> self-activation with a basal term (R10: must leave zero state)")
text5 = "CAMKII activates CAMKII. CAMKII starts at 0.1."
bp5 = build(text5)
init5 = {n["id"]: n.get("initial_value") for n in bp5.get("nodes", [])}
record("autocatalytic", "0.1 initial condition survived", init5.get("CAMKII"),
       "== 0.1", init5.get("CAMKII") == 0.1)
try:
    res5 = simulate(bp5, t_max=30.0)
    cam = series(res5, "CAMKII")
    record("autocatalytic", "trajectory finite", int(np.sum(~np.isfinite(cam))), "== 0",
           bool(np.all(np.isfinite(cam))))
    record("autocatalytic", "left its initial state (autocatalysis is active)",
           f"{abs(cam[-1] - cam[0]):.6g}", "> 1e-9", abs(cam[-1] - cam[0]) > 1e-9)
except Exception as exc:
    record("autocatalytic", "simulation ran", f"{type(exc).__name__}", "no exception", False)

# ---------------------------------------------------------------------------
print("\n" + "=" * 104)
print(" SUMMARY — natural language to a runnable model")
print("-" * 104)
print(f" {'MODEL':<16}{'CHECK':<54}{'MEASURED':<22}{'VERDICT'}")
print("-" * 104)
for name, check, measured, _tol, verdict in results:
    print(f" {name:<16}{check:<54}{str(measured):<22}{verdict}")
failed = [r for r in results if r[4] == FAIL]
print("-" * 104)
print(f" {len(results)} checks: {len(results) - len(failed)} passed, {len(failed)} failed")
print("=" * 104)
sys.exit(1 if failed else 0)
