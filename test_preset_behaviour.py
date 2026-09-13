"""
BEHAVIOURAL TEST TIER FOR THE SHIPPED MODEL PRESETS
===================================================

Every other test module in this repo tests PLUMBING: does the endpoint return 200,
does the project round-trip, does the state machine refuse an illegal transition.
None of them asks the only question a researcher cares about when they click a
preset button: **does the oscillator oscillate?**

This module asks exactly that, for all eight shipped presets, numerically:

    EGF/EGFR .......... the readout named in the preset's own targets must PEAK
                        before t_max and then decay (a transient MAPK response)
    Turing (PDE) ...... spatial pattern amplitude must GROW, every frame finite
    Oscillator ........ >= 4 peaks, a measurable regular period, and an amplitude
                        in the final fifth >= 80% of the second fifth
    Bistable .......... two initial conditions (0 and 10) must stay separated at
                        t_max AND at 10*t_max
    Fold-change ....... the peak response to a fixed ~3x input step must agree
                        within 15% across a 100x span of absolute input levels,
                        sampled at 5 levels
    Berridge & Goldbeter 1990 ... sustained cytosolic Ca2+ oscillations
    Zhabotinsky 2000 ............ a transient Ca2+ pulse latches CaMKII ON and it
                                  stays ON; the OFF state is stable without a pulse
    Lyashenko 2020 .............. fold-change detection: equal fold steps give
                                  equal response pulses, at any absolute level

These assertions state what each preset's NAME PROMISES, not what the code
currently does. **Do not weaken a tolerance to make a test pass** -- fix the preset.

Two-level tolerance policy: >= 4 peaks and a >= 80% amplitude ratio are the
qualitative bar (oscillating at all vs. ringing down); 15% agreement is the
quantitative bar for the two fold-change presets, because Weber's law is a claim
about numbers matching, not about a trend.

STATUS as measured on 2026-09-13 (python 3.13, scipy LSODA, rtol=1e-8/atol=1e-10)
---------------------------------------------------------------------------------
    FAIL EGF/EGFR    : ERK peak_time = 50.00 == t_max = 50.0. The readout is still
                       rising when the run ends (it only turns over at t = 56.2),
                       so final/peak = 1.000 -- no peak and no decay inside the
                       shipped horizon, and the preset's own "ERK peaks at 5-15
                       min" / "peak value 0.6-1.0" targets read 50.00 and 4.709.
    PASS Oscillator  : 11 peaks over t_max = 300, period 25.98 (std/mean 0.0013),
                       amplitude ratio final/second fifth = 1.000003
    PASS Bistable    : OFF 0.040005 / ON 1.896486, relative separation 0.9789 --
                       bit-identical at t = 100 and t = 1000
    PASS Fold-change : responses 1.4101 / 1.4090 / 1.4087 / 1.4085 / 1.4085 to the
                       same 3x step at ambient levels 0.3 / 1 / 3 / 10 / 30
                       -> spread 0.11%
    PASS Turing, Berridge, Zhabotinsky, Lyashenko

HOW THE BLUEPRINTS GET HERE
---------------------------
The preset definitions live in ``static/app.js``. They are FROZEN into this file
as plain dicts so the tests never parse JavaScript at run time:

  * egfr, oscillator, bistable, foldchange (Presets) and berridge, zhabotinsky,
    lyashenko (PaperModels) ship an explicit ``blueprint`` object in app.js --
    copied verbatim below.
  * turing ships only ``text`` + ``targets``; the blueprint the user actually runs
    is what ``agent.rule_based_parse(text)`` returns (the deterministic, LLM-off
    path behind ``POST /api/blueprint``). That output is frozen below.

If a preset's shipped definition changes, REFRESH THE FROZEN COPY HERE, or these
tests will keep measuring the old model. ``PresetDefinitionDriftTest`` exists to
make that impossible to miss: it fails when a frozen preset text, a frozen ODE /
flux expression, or a frozen parameter value no longer appears in app.js. (It only
searches app.js for strings -- it never builds a blueprint out of JavaScript.)

Run just this tier while iterating:

    python -m unittest test_preset_behaviour
"""

import os
import re
import unittest

import numpy as np
from scipy.signal import find_peaks

import agent
from simulation_engine import ODEModel, solve_pde

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
APP_JS = os.path.join(REPO_DIR, "static", "app.js")

# Deterministic noise for the PDE preset (solve_pde seeds its grids from numpy's
# global RNG, so an unseeded run would make the Turing assertions flaky).
PDE_SEED = 20260913


# ==========================================================================
# FROZEN PRESET DEFINITIONS  (static/app.js, 2026-09-13)
# ==========================================================================

# --- Turing: the only text-only preset; blueprint below is what
# --- agent.rule_based_parse() deterministically compiles the text to -----------

TURING_TEXT = """A reaction-diffusion system containing Activator U and Inhibitor V.
U activates itself and activates V.
V inhibits U.
U starts at 1.0.
V starts at 1.0.
U diffuses slowly, V diffuses quickly."""

_HILL_PDE = {"k": 1.0, "K_d": 1.0, "n": 2.0}

TURING_BLUEPRINT = {
    "type": "PDE",
    "nodes": [
        {"id": "U", "name": "Activator U", "initial_value": 1.0},
        {"id": "V", "name": "Inhibitor V", "initial_value": 1.0},
    ],
    "edges": [
        {"source": "U", "target": "U", "type": "activation", "parameters": dict(_HILL_PDE)},
        {"source": "U", "target": "V", "type": "activation", "parameters": dict(_HILL_PDE)},
        {"source": "V", "target": "U", "type": "inhibition", "parameters": dict(_HILL_PDE)},
    ],
    "spatial": {
        "x_grid": 50, "y_grid": 50, "dx": 1.0, "dy": 1.0,
        "diffusion": {"U": 0.05, "V": 1.0},
        "reactions": {"U": "U**2 / V - U + 0.02", "V": "U**2 - V"},
    },
    "simulation_config": {"t_max": 200.0, "dt": 0.1},
}

# --- EGF/EGFR ----------------------------------------------------------------

EGFR_TEXT = """EGF binds to EGFR and activates it.
EGFR activates RAS.
RAS activates RAF.
RAF activates MEK.
MEK activates ERK.
ERK inhibits EGFR.
EGF starts at 10.0.
EGFR starts at 1.0.
RAS starts at 1.0.
RAF starts at 1.0."""

EGFR_BLUEPRINT = {
    "type": "ODE",
    "nodes": [
        {"id": "EGF", "initial_value": 10.0}, {"id": "EGFR", "initial_value": 1.0},
        {"id": "RAS", "initial_value": 1.0}, {"id": "RAF", "initial_value": 1.0},
        {"id": "MEK", "initial_value": 0.0}, {"id": "ERK", "initial_value": 0.0},
    ],
    "edges": [
        {"source": "EGF", "target": "EGFR", "type": "activation"},
        {"source": "EGFR", "target": "RAS", "type": "activation"},
        {"source": "RAS", "target": "RAF", "type": "activation"},
        {"source": "RAF", "target": "MEK", "type": "activation"},
        {"source": "MEK", "target": "ERK", "type": "activation"},
        {"source": "ERK", "target": "EGFR", "type": "inhibition"},
    ],
    # Explicit rate laws, mirroring app.js. Nodes and edges alone left this to the
    # generic Hill compiler, under which ERK rose monotonically to 4.71 with its
    # maximum AT t_max -- no peak, no adaptation, and 2 of the preset's own 3 targets
    # failing on the preset that auto-loads on page open. Two things were missing:
    # a dephosphorylation term on every stage (without one a species can only
    # accumulate) and a transient stimulus (EGF was pinned at 10.0 for ever, so the
    # receptor was continuously re-driven despite the ERK feedback edge).
    "parameters": {
        "kEGF": 0.8,
        "kR": 1.8, "KmR": 5.0,
        "KiR": 0.06, "nR": 3.0,
        "dR": 0.25,
        "k1": 0.9, "k2": 0.9, "k3": 0.9, "k4": 0.9,
        "d1": 0.35, "d2": 0.35, "d3": 0.35, "d4": 0.35,
    },
    "odes": {
        "EGF": "-kEGF*EGF*EGFR",
        "EGFR": "kR*EGF/(KmR + EGF)*(1 - EGFR)/(1 + (ERK/KiR)**nR) - dR*EGFR",
        "RAS": "k1*EGFR*(1 - RAS) - d1*RAS",
        "RAF": "k2*RAS*(1 - RAF) - d2*RAF",
        "MEK": "k3*RAF*(1 - MEK) - d3*MEK",
        "ERK": "k4*MEK*(1 - ERK) - d4*ERK",
    },
    "simulation_config": {"t_max": 50.0},
}
# The preset's OWN targets, verbatim from app.js. The readout under test is the
# species its peak_time target names, i.e. ERK.
EGFR_TARGETS = [
    {"species": "ERK", "type": "peak_time", "min": 5.0, "max": 15.0},
    {"species": "ERK", "type": "peak_value", "min": 0.6, "max": 1.0},
    {"species": "EGFR", "type": "decay_ratio", "max": 0.2},
]
EGFR_READOUT = "ERK"

# --- Oscillator: Goodwin (1965) three-stage negative feedback, Hill n = 16 ----

OSCILLATOR_TEXT = """GENA activates GENB.
GENB activates GENC.
GENC inhibits GENA.
GENA starts at 2.851.
GENB starts at 1.495.
GENC starts at 1.024."""

OSCILLATOR_BLUEPRINT = {
    "type": "ODE",
    "name": "Goodwin three-stage negative-feedback oscillator",
    "nodes": [
        {"id": "GENA", "name": "Gene A product", "initial_value": 2.851},
        {"id": "GENB", "name": "Gene B product", "initial_value": 1.495},
        {"id": "GENC", "name": "Gene C product (repressor)", "initial_value": 1.024},
    ],
    "edges": [
        {"id": "e1", "source": "GENA", "target": "GENB", "type": "activation"},
        {"id": "e2", "source": "GENB", "target": "GENC", "type": "activation"},
        {"id": "e3", "source": "GENC", "target": "GENA", "type": "inhibition"},
    ],
    "parameters": {
        "v1": 1.0, "K1": 1.0, "n": 16.0,
        "d1": 0.15, "k3": 0.15, "d2": 0.15, "k5": 0.15, "d3": 0.15,
    },
    "odes": {
        "GENA": "v1*K1**n/(K1**n + GENC**n) - d1*GENA",
        "GENB": "k3*GENA - d2*GENB",
        "GENC": "k5*GENB - d3*GENC",
    },
    "plot_species": ["GENA", "GENB", "GENC"],
    "simulation_config": {"t_max": 300.0},
}
OSCILLATOR_READOUT = "GENA"          # the species the preset's oscillation target names

# --- Bistable: cooperative positive autofeedback + first-order removal --------

BISTABLE_TEXT = """STIM activates CAMKII.
CAMKII activates itself.
STIM starts at 1.0.
CAMKII starts at 0.1."""

BISTABLE_BLUEPRINT = {
    "type": "ODE",
    "name": "Cooperative positive-feedback bistable switch",
    "nodes": [
        {"id": "STIM", "name": "Stimulus", "initial_value": 1.0},
        {"id": "CAMKII", "name": "Active CaMKII", "initial_value": 0.1},
    ],
    "edges": [
        {"id": "e1", "source": "STIM", "target": "CAMKII", "type": "activation"},
        {"id": "e2", "source": "CAMKII", "target": "CAMKII", "type": "activation"},
    ],
    "parameters": {
        "ks": 1.0, "Sset": 1.0, "kbas": 0.02,
        "kfb": 1.0, "Kfb": 1.0, "n": 4.0, "kdeg": 0.5,
    },
    "odes": {
        "STIM": "ks*(Sset - STIM)",
        "CAMKII": "kbas*STIM + kfb*CAMKII**n/(Kfb**n + CAMKII**n) - kdeg*CAMKII",
    },
    "plot_species": ["STIM", "CAMKII"],
    "simulation_config": {"t_max": 100.0},
}
BISTABLE_READOUT = "CAMKII"

# --- Fold-change: incoherent feed-forward loop (Goentoro & Alon 2009) ---------

FOLDCHANGE_TEXT = """EGF activates AKT.
EGF activates BG.
BG inhibits AKT.
EGF starts at 1.0.
BG starts at 500.0.
AKT starts at 1.0."""

FOLDCHANGE_BLUEPRINT = {
    "type": "ODE",
    "name": "Fold-change detection (incoherent feed-forward loop)",
    "nodes": [
        {"id": "EGF", "name": "Ambient EGF level", "initial_value": 1.0},
        {"id": "BG", "name": "Adapted EGF background", "initial_value": 500.0},
        {"id": "AKT", "name": "Relative AKT response", "initial_value": 1.0},
    ],
    "edges": [
        {"id": "e1", "source": "EGF", "target": "BG", "type": "activation"},
        {"id": "e2", "source": "EGF", "target": "AKT", "type": "activation"},
        {"id": "e3", "source": "BG", "target": "AKT", "type": "inhibition"},
    ],
    "parameters": {"a": 0.2, "kf": 8.0, "fold": 3.0, "sr": 6.0, "t_on": 70.0},
    "fluxes": {
        "step": "1 + (fold-1)/(1+exp(-sr*(t-t_on)))",
        "Lig": "EGF*step",
    },
    "odes": {
        "EGF": "0",
        "BG": "a*(Lig - BG)",
        "AKT": "kf*(Lig/BG - AKT)",
    },
    "plot_species": ["EGF", "BG", "AKT"],
    "simulation_config": {"t_max": 140.0},
}
FOLDCHANGE_INPUT, FOLDCHANGE_OUTPUT = "EGF", "AKT"

# --- Berridge & Goldbeter (1990) ---------------------------------------------

BERRIDGE_BLUEPRINT = {
    "type": "ODE",
    "name": "Berridge-Goldbeter Ca2+ oscillator",
    "nodes": [
        {"id": "Z", "name": "Cytosolic Ca2+", "initial_value": 0.1},
        {"id": "Y", "name": "Internal-store Ca2+", "initial_value": 0.1},
    ],
    "edges": [
        {"id": "e1", "source": "Z", "target": "Y", "type": "activation"},
        {"id": "e2", "source": "Y", "target": "Z", "type": "activation"},
    ],
    "parameters": {
        "v0": 1.0, "v1": 7.3, "beta": 0.5, "VM2": 65.0, "VM3": 500.0,
        "K2": 1.0, "KR": 2.0, "KA": 0.9, "kf": 1.0, "k": 10.0,
    },
    "fluxes": {
        "v2": "VM2*Z**2/(K2**2+Z**2)",
        "v3": "VM3*(Y**2/(KR**2+Y**2))*(Z**4/(KA**4+Z**4))",
    },
    "odes": {
        "Z": "v0 + v1*beta - v2 + v3 + kf*Y - k*Z",
        "Y": "v2 - v3 - kf*Y",
    },
    "plot_species": ["Z", "Y"],
    "simulation_config": {"t_max": 10.0},
}
BERRIDGE_READOUT = "Z"               # cytosolic Ca2+

# --- Zhabotinsky (2000) ------------------------------------------------------

ZHABOTINSKY_BLUEPRINT = {
    "type": "ODE",
    "name": "Zhabotinsky CaMKII bistable switch",
    "nodes": (
        [{"id": "Ca", "name": "Calcium stimulus", "initial_value": 2.0},
         {"id": "P0", "name": "Unphosphorylated CaMKII", "initial_value": 2.0}]
        + [{"id": "P%d" % i, "initial_value": 0.0} for i in range(1, 11)]
        + [{"id": "A", "name": "Active CaMKII", "initial_value": 0.0}]
    ),
    "edges": [
        {"id": "e1", "source": "Ca", "target": "A", "type": "activation"},
        {"id": "e2", "source": "A", "target": "A", "type": "activation"},
    ],
    "parameters": {
        "k1": 0.5, "k2": 2.0, "KH1": 4.0, "KM": 0.4, "ep": 0.05,
        "kca": 5.0, "Cabase": 2.0, "amp": 1.0, "sr": 2.0,
        "t_on": 20.0, "t_off": 60.0, "kobs": 50.0,
    },
    "fluxes": {
        "v1": "10*k1*(Ca/KH1)**8*P0/(1 + (Ca/KH1)**4)**2",
        "v2": "k1*(Ca/KH1)**4/(1 + (Ca/KH1)**4)",
        "Ssum": "1*P1+2*P2+3*P3+4*P4+5*P5+6*P6+7*P7+8*P8+9*P9+10*P10",
        "v3": "k2*ep/(KM + Ssum)",
    },
    "odes": {
        "P0": "-v1 + v3*1*P1",
        "P1": "v1 - v2*1.0*P1 - v3*1*P1 + v3*2*P2",
        "P2": "v2*1.0*P1 - v2*1.8*P2 - v3*2*P2 + v3*3*P3",
        "P3": "v2*1.8*P2 - v2*2.3*P3 - v3*3*P3 + v3*4*P4",
        "P4": "v2*2.3*P3 - v2*2.7*P4 - v3*4*P4 + v3*5*P5",
        "P5": "v2*2.7*P4 - v2*2.8*P5 - v3*5*P5 + v3*6*P6",
        "P6": "v2*2.8*P5 - v2*2.7*P6 - v3*6*P6 + v3*7*P7",
        "P7": "v2*2.7*P6 - v2*2.3*P7 - v3*7*P7 + v3*8*P8",
        "P8": "v2*2.3*P7 - v2*1.8*P8 - v3*8*P8 + v3*9*P9",
        "P9": "v2*1.8*P8 - v2*1.0*P9 - v3*9*P9 + v3*10*P10",
        "P10": "v2*1.0*P9 - v3*10*P10",
        "Ca": "kca*(Cabase + amp/(1+exp(-sr*(t-t_on))) - amp/(1+exp(-sr*(t-t_off))) - Ca)",
        "A": "kobs*(Ssum - A)",
    },
    "plot_species": ["Ca", "A"],
    "simulation_config": {"t_max": 350.0},
}
ZHABOTINSKY_READOUT = "A"            # active CaMKII
ZHAB_PULSE_ON, ZHAB_PULSE_OFF = 20.0, 60.0

# --- Lyashenko et al. (2020) -------------------------------------------------

LYASHENKO_BLUEPRINT = {
    "type": "ODE",
    "name": "Lyashenko fold-change detection",
    "nodes": [
        {"id": "L", "name": "Ligand", "initial_value": 1.0},
        {"id": "R", "name": "Adapted background (receptor memory)", "initial_value": 1.0},
        {"id": "S", "name": "Relative response", "initial_value": 1.0},
    ],
    "edges": [
        {"id": "e1", "source": "L", "target": "R", "type": "activation"},
        {"id": "e2", "source": "L", "target": "S", "type": "activation"},
        {"id": "e3", "source": "R", "target": "S", "type": "inhibition"},
    ],
    "parameters": {
        "a": 0.2, "kf": 20.0, "kl": 40.0, "sr": 8.0,
        "L1": 1.0, "L2": 2.0, "L3": 4.0, "L4": 8.0, "L5": 16.0,
        "t1": 20.0, "t2": 45.0, "t3": 70.0, "t4": 95.0,
    },
    "odes": {
        "L": ("kl*(L1 + (L2-L1)/(1+exp(-sr*(t-t1))) + (L3-L2)/(1+exp(-sr*(t-t2)))"
              " + (L4-L3)/(1+exp(-sr*(t-t3))) + (L5-L4)/(1+exp(-sr*(t-t4))) - L)"),
        "R": "a*(L - R)",
        "S": "kf*(L/R - S)",
    },
    "plot_species": ["L", "R", "S"],
    "simulation_config": {"t_max": 125.0},
}
LYASHENKO_READOUT = "S"
LYASHENKO_LEVELS = [1.0, 2.0, 4.0, 8.0, 16.0]     # the shipped 2x staircase
LYASHENKO_BASELINE = 1.0                          # S re-adapts to L/R -> 1

# Every frozen blueprint, for the drift guard.
FROZEN_BLUEPRINTS = {
    "egfr": EGFR_BLUEPRINT,
    "turing": TURING_BLUEPRINT,
    "oscillator": OSCILLATOR_BLUEPRINT,
    "bistable": BISTABLE_BLUEPRINT,
    "foldchange": FOLDCHANGE_BLUEPRINT,
    "berridge": BERRIDGE_BLUEPRINT,
    "zhabotinsky": ZHABOTINSKY_BLUEPRINT,
    "lyashenko": LYASHENKO_BLUEPRINT,
}
FROZEN_TEXTS = {
    "egfr": EGFR_TEXT,
    "turing": TURING_TEXT,
    "oscillator": OSCILLATOR_TEXT,
    "bistable": BISTABLE_TEXT,
    "foldchange": FOLDCHANGE_TEXT,
}


# ==========================================================================
# MEASUREMENT HELPERS
# ==========================================================================

def app_num_points(t_max):
    """The resolution the shipped POST /api/simulate uses for a given horizon."""
    return int(min(5000, max(300, t_max * 15)))


def simulate(blueprint, t_max=None, num_points=None, custom_params=None, custom_initial=None):
    """ODEModel(blueprint).simulate(...) at the app's resolution, with a readable
    failure if the integrator gives up.

    Overflow in exp() is ignored because several shipped presets drive a stimulus
    with logistic terms (``1/(1+exp(-sr*(t-t_on)))``), where exp overflows to inf and
    the term correctly evaluates to 0. That cannot hide a diverging run: every
    returned trajectory is checked for finiteness here.
    """
    t_max = float(t_max if t_max is not None
                  else blueprint.get("simulation_config", {}).get("t_max", 50.0))
    model = ODEModel(blueprint)
    try:
        with np.errstate(over="ignore"):
            res = model.simulate(t_max,
                                 num_points=num_points or app_num_points(t_max),
                                 custom_params=custom_params,
                                 custom_initial=custom_initial)
    except RuntimeError as exc:                    # integrator refused to converge
        raise AssertionError(
            "the preset could not even be integrated to t_max=%g: %s" % (t_max, exc)) from exc
    species = {k: np.asarray(v, dtype=float) for k, v in res["species"].items()}
    for name, values in species.items():
        if not np.all(np.isfinite(values)):
            raise AssertionError(
                "species %s contains non-finite values over t_max=%g (min %.4g, max %.4g): "
                "the trajectory diverged." % (name, t_max, np.nanmin(values), np.nanmax(values)))
    return np.asarray(res["t"], dtype=float), species


def peak_indices(y, prominence_frac=0.05):
    """Indices of local maxima whose topographic prominence is at least
    ``prominence_frac`` of the trajectory's full range.

    Prominence (not a neighbour-difference test) is what makes this independent of
    sampling density: a smooth peak sampled 10x more finely has ~100x smaller
    neighbour differences but the same prominence, so a naive detector silently
    reports zero peaks on a densely sampled limit cycle.
    """
    y = np.asarray(y, dtype=float)
    if y.size < 5 or not np.all(np.isfinite(y)):
        return np.array([], dtype=int)
    rng = float(y.max() - y.min())
    if rng < 1e-12:
        return np.array([], dtype=int)
    idx, _ = find_peaks(y, prominence=prominence_frac * rng)
    return idx


def fifth_amplitude(y, fifth):
    """Peak-to-trough amplitude inside the ``fifth``-th (1-based) fifth of a run."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    seg = y[(fifth - 1) * n // 5: fifth * n // 5]
    if seg.size == 0:
        return 0.0
    return float(seg.max() - seg.min())


def period_stats(t, y, after=None, prominence_frac=0.05):
    """(peak_times, mean_period, coefficient_of_variation) for a trajectory."""
    idx = peak_indices(y, prominence_frac)
    times = np.asarray(t, dtype=float)[idx]
    if after is not None:
        times = times[times >= after]
    if times.size < 2:
        return times, None, None
    intervals = np.diff(times)
    mean = float(intervals.mean())
    cv = float(intervals.std() / mean) if mean > 0 else None
    return times, mean, cv


def relative_separation(low_final, high_final):
    """Separation of two attractors, normalised the way agent.bistability_separation
    normalises it, so the number is directly comparable to the preset's own target."""
    return abs(high_final - low_final) / max(abs(high_final), abs(low_final), 1.0)


def bistable_finals(blueprint, species, t_max, low=0.0, high=10.0):
    """Final readout value settling from a LOW and a HIGH initial condition."""
    _, lo = simulate(blueprint, t_max=t_max, num_points=400, custom_initial={species: low})
    _, hi = simulate(blueprint, t_max=t_max, num_points=400, custom_initial={species: high})
    return float(lo[species][-1]), float(hi[species][-1])


def value_at(t, y, when):
    return float(np.interp(when, np.asarray(t, dtype=float), np.asarray(y, dtype=float)))


def has_internal_step(blueprint):
    """True when the blueprint applies its own fold-change step in the equations
    (parameters ``fold`` + ``t_on``), as the shipped fold-change preset does."""
    params = blueprint.get("parameters") or {}
    return "fold" in params and "t_on" in params and bool(blueprint.get("odes"))


def fold_change_response(blueprint, input_species, output_species, base_level,
                         t_max=None, fold=3.0):
    """Relative peak response of the output to a ``fold`` step of the input, with the
    system ADAPTED to ``base_level`` beforehand. Returns (response, baseline, peak).

    Two protocols, chosen from the blueprint's own shape:

    * internal step (the shipped preset): the model applies the fold change itself at
      ``t_on``, and the input species is the clamped ambient level. One run per level:
      set the ambient level, read the output just before ``t_on`` as the adapted
      baseline, and take the maximum after ``t_on`` as the peak.
    * step-and-hold (a generic Hill blueprint with no internal step): hold the input
      at ``base_level`` for 3*t_max so everything adapts, then re-run with the input
      stepped to ``base_level * fold``, every other species starting from its adapted
      value. The input is held by setting its basal synthesis to deg_<input> * level,
      which is the generic compiler's own way of pinning a species' steady state.

    Either way, response = (peak - adapted baseline) / adapted baseline, i.e. the
    relative overshoot -- the quantity Weber's law says must not depend on the level.
    """
    config = blueprint.get("simulation_config", {})
    t_max = float(t_max if t_max is not None else config.get("t_max", 60.0))

    if has_internal_step(blueprint):
        params = blueprint["parameters"]
        t_on = float(params["t_on"])
        initial = {node["id"]: float(node.get("initial_value", 0.0))
                   for node in blueprint["nodes"]}
        initial[input_species] = base_level
        t, species = simulate(blueprint, t_max=t_max, custom_initial=initial)
        y = species[output_species]
        baseline = value_at(t, y, max(t_on - 1.0, 0.0))
        after = y[t >= t_on]
        peak = float(after.max()) if after.size else float(y.max())
    else:
        model = ODEModel(blueprint)
        syn, deg = "syn_%s" % input_species, "deg_%s" % input_species
        clampable = syn in model.params_dict and deg in model.params_dict
        deg_val = float(model.params_dict.get(deg, 0.1)) if clampable else 0.0

        def hold(level):
            return {syn: deg_val * level} if clampable else None

        _, pre = simulate(blueprint, t_max=3.0 * t_max, num_points=400,
                          custom_params=hold(base_level),
                          custom_initial={input_species: base_level})
        adapted = {name: float(vals[-1]) for name, vals in pre.items()}
        stepped = dict(adapted)
        stepped[input_species] = base_level * fold
        _, post = simulate(blueprint, t_max=t_max,
                           custom_params=hold(base_level * fold),
                           custom_initial=stepped)
        y = post[output_species]
        baseline = adapted[output_species]
        peak = float(y.max())

    if abs(baseline) < 1e-12:
        return float("inf"), baseline, peak
    return (peak - baseline) / abs(baseline), baseline, peak


def run_turing_pde(blueprint, t_max=None, seed=PDE_SEED):
    """Run the PDE preset exactly the way POST /api/simulate runs it.

    solve_pde seeds its grids from numpy's global RNG, so the seed is pinned for
    reproducibility -- and the previous global state is restored afterwards, so this
    module cannot change the random stream any other test module sees.
    """
    spatial = blueprint["spatial"]
    config = blueprint.get("simulation_config", {})
    initial = {node["id"]: {"type": "random_noise",
                            "base_value": node.get("initial_value", 1.0),
                            "noise_amplitude": 0.05}
               for node in blueprint["nodes"]}
    rng_state = np.random.get_state()
    np.random.seed(seed)
    try:
        return solve_pde(spatial_config=spatial,
                         reaction_formulas=spatial["reactions"],
                         initial_conditions=initial,
                         t_max=float(t_max if t_max is not None
                                     else config.get("t_max", 200.0)),
                         dt=float(config.get("dt", 0.1)),
                         save_every=5)
    finally:
        np.random.set_state(rng_state)


def frame_stds(frames):
    """Spatial standard deviation per saved frame = pattern amplitude over time."""
    return np.array([float(np.std(np.asarray(f, dtype=float))) for f in frames])


# ==========================================================================
# EGF/EGFR
# ==========================================================================

class EgfrPresetBehaviourTest(unittest.TestCase):
    """EGF/EGFR preset: a transient MAPK response -- ERK rises, peaks and decays,
    and the receptor is downregulated by the ERK feedback."""

    @classmethod
    def setUpClass(cls):
        cls.t, cls.sp = simulate(EGFR_BLUEPRINT)
        cls.t_max = EGFR_BLUEPRINT["simulation_config"]["t_max"]

    def test_readout_peaks_before_t_max_and_then_decays(self):
        """EXPECTED FAIL until the EGF/EGFR preset is retuned.

        Measured 2026-09-13: ERK peak_time = 50.00, which IS t_max = 50.0 -- ERK is
        still rising when the run ends (it only turns over at t = 56.2), so
        final/peak = 1.000 instead of decaying. A readout whose maximum is its last
        sample has not been shown to peak at all.
        """
        y = self.sp[EGFR_READOUT]
        peak_idx = int(np.argmax(y))
        peak_time = float(self.t[peak_idx])
        peak_val = float(y.max())
        final = float(y[-1])
        decay_ratio = final / peak_val if peak_val > 0 else float("inf")

        self.assertGreater(peak_val, 1e-6,
                           "%s never activates: peak value %.3g" % (EGFR_READOUT, peak_val))
        self.assertLess(
            peak_time, self.t_max,
            "%s does not peak inside the shipped horizon: peak_time = %.2f vs t_max = %.2f. "
            "The maximum being the final sample means the readout is still rising when the "
            "run ends, so the response is not transient." % (EGFR_READOUT, peak_time, self.t_max))
        self.assertLessEqual(
            decay_ratio, 0.8,
            "%s does not decay after its peak: final/peak = %.3f (required <= 0.80, i.e. at "
            "least a 20%% fall from the peak by t_max). peak = %.4f at t = %.2f, final = %.4f."
            % (EGFR_READOUT, decay_ratio, peak_val, peak_time, final))

    def test_receptor_is_downregulated_by_feedback(self):
        """EGFR must be driven back down by the ERK -> EGFR inhibition: the preset's
        own decay_ratio target is final/peak <= 0.20. Measured: 0.028 (passes)."""
        y = self.sp["EGFR"]
        peak_val, final = float(y.max()), float(y[-1])
        ratio = final / peak_val if peak_val > 0 else float("inf")
        self.assertLessEqual(
            ratio, 0.2,
            "EGFR is not downregulated: final/peak = %.3f (preset target <= 0.20). "
            "peak = %.4f, final = %.4f -- the ERK -> EGFR negative feedback is not closing."
            % (ratio, peak_val, final))

    def test_preset_meets_its_own_declared_targets(self):
        """EXPECTED FAIL until the EGF/EGFR preset is retuned.

        The preset ships three targets; a plain "Run Simulation" should satisfy them
        with no optimizer pass. Measured 2026-09-13: the ERK peak_time target
        (5-15 min) reads 50.00 min and the ERK peak_value target (0.6-1.0) reads
        4.709. Only the EGFR decay_ratio target passes.
        """
        failures = []
        for target in EGFR_TARGETS:
            species = target["species"]
            ok, message = agent.TargetMetric(target).evaluate(
                self.t.tolist(), self.sp[species].tolist())
            if not ok:
                failures.append("%s/%s: %s" % (species, target["type"], message))
        self.assertEqual(
            [], failures,
            "the EGF/EGFR preset does not meet the targets it ships with:\n  "
            + "\n  ".join(failures))


# ==========================================================================
# TURING (PDE)
# ==========================================================================

class TuringPresetBehaviourTest(unittest.TestCase):
    """Turing preset: a homogeneous field plus noise must develop a spatial pattern
    (amplitude growth), in a numerically clean run."""

    @classmethod
    def setUpClass(cls):
        cls.result = run_turing_pde(TURING_BLUEPRINT)

    def test_spatial_pattern_amplitude_grows(self):
        """Pattern amplitude = spatial std per frame. Required: at least 5x growth for
        the activator and a final amplitude >= 0.05 (a pattern, not residual noise).
        Measured: std 0.0147 -> 0.920, i.e. 62.4x (passes)."""
        stds = frame_stds(self.result["species"]["U"])
        self.assertGreaterEqual(len(stds), 5,
                                "only %d frames were saved; cannot judge growth" % len(stds))
        first, last = float(stds[0]), float(stds[-1])
        growth = last / max(first, 1e-12)
        self.assertGreaterEqual(
            growth, 5.0,
            "no Turing pattern formed in U: spatial std went %.5f -> %.5f (%.2fx, required "
            ">= 5x). A flat field means the instability never grew." % (first, last, growth))
        self.assertGreaterEqual(
            last, 0.05,
            "U's final pattern amplitude is only %.5f (required >= 0.05): the field is still "
            "essentially the initial noise (%.5f)." % (last, first))

    def test_every_frame_is_finite_and_run_did_not_diverge(self):
        """An explicit forward-Euler PDE that diverges can still look plausible once
        negatives are clamped, so assert finiteness frame by frame and check the
        solver's own stability report. Measured: diffusion number 0.2 (<= 0.5),
        diverged_at = None (passes)."""
        for species, frames in self.result["species"].items():
            for i, frame in enumerate(frames):
                arr = np.asarray(frame, dtype=float)
                self.assertTrue(
                    np.all(np.isfinite(arr)),
                    "species %s frame %d/%d contains non-finite values (min %.4g, max %.4g): "
                    "the explicit scheme diverged."
                    % (species, i, len(frames), np.nanmin(arr), np.nanmax(arr)))
        stability = self.result["stability"]
        self.assertIsNone(
            stability["diverged_at"],
            "solve_pde reported divergence at t = %s (diffusion number %.3f, max stable dt %s)"
            % (stability["diverged_at"], stability["diffusion_number"],
               stability["max_stable_dt"]))
        self.assertLessEqual(
            stability["diffusion_number"], 0.5 + 1e-12,
            "the shipped dt violates the explicit-scheme stability bound: diffusion number "
            "%.3f > 0.5 (max stable dt %s, shipped dt %s)"
            % (stability["diffusion_number"], stability["max_stable_dt"], stability["dt"]))


# ==========================================================================
# OSCILLATOR
# ==========================================================================

class OscillatorPresetBehaviourTest(unittest.TestCase):
    """Oscillator preset: the question this whole tier exists for -- does the
    oscillator oscillate, and does it KEEP oscillating?"""

    @classmethod
    def setUpClass(cls):
        cls.t_max = OSCILLATOR_BLUEPRINT["simulation_config"]["t_max"]
        cls.t, cls.sp = simulate(OSCILLATOR_BLUEPRINT, t_max=cls.t_max)
        cls.y = cls.sp[OSCILLATOR_READOUT]

    def test_produces_at_least_four_peaks(self):
        """Required: >= 4 prominent peaks over the preset's own t_max = 300.
        Measured: 11 peaks, range 0.3089 .. 2.8514 (passes)."""
        peaks = peak_indices(self.y)
        self.assertGreaterEqual(
            len(peaks), 4,
            "%s shows %d prominent peaks over t_max = %g (required >= 4). Trajectory range "
            "%.4g .. %.4g, final value %.4g -- an oscillator that never comes back up is a "
            "transient." % (OSCILLATOR_READOUT, len(peaks), self.t_max,
                            float(self.y.min()), float(self.y.max()), float(self.y[-1])))

    def test_period_is_measurable_and_regular(self):
        """Required: >= 3 peaks in the second half of the run, a positive mean
        inter-peak interval, and interval variability (std/mean) <= 0.25 -- a period
        you could quote in a paper. Measured: 6 peaks after t = 150, period 25.98,
        std/mean 0.0013 (passes)."""
        times, mean_period, cv = period_stats(self.t, self.y, after=self.t_max / 2.0)
        self.assertGreaterEqual(
            len(times), 3,
            "cannot measure a period for %s: only %d peaks after t = %g (required >= 3). "
            "%d peaks over the whole run."
            % (OSCILLATOR_READOUT, len(times), self.t_max / 2.0, len(peak_indices(self.y))))
        self.assertIsNotNone(mean_period, "no inter-peak interval could be computed")
        self.assertGreater(mean_period, 0.0,
                           "mean period is %.4g, which is not a period" % mean_period)
        self.assertLessEqual(
            cv, 0.25,
            "%s's period is not regular: mean %.3f, std/mean %.3f (required <= 0.25). "
            "Intervals: %s" % (OSCILLATOR_READOUT, mean_period, cv,
                               np.round(np.diff(times), 3).tolist()))

    def test_amplitude_is_sustained_not_decaying(self):
        """Required: amplitude in the FINAL fifth >= 80% of the amplitude in the SECOND
        fifth, which separates a limit cycle from a damped ringdown. Measured: 2.5425
        vs 2.5425, ratio 1.000003 (passes)."""
        early = fifth_amplitude(self.y, 2)
        late = fifth_amplitude(self.y, 5)
        self.assertGreater(
            early, 1e-9,
            "%s is flat even in the second fifth of the run (amplitude %.3g): nothing to "
            "sustain." % (OSCILLATOR_READOUT, early))
        ratio = late / early
        self.assertGreaterEqual(
            ratio, 0.8,
            "%s's oscillation is decaying, not sustained: amplitude %.4g in the final fifth "
            "vs %.4g in the second fifth (ratio %.4g, required >= 0.80)."
            % (OSCILLATOR_READOUT, late, early, ratio))


# ==========================================================================
# BISTABLE
# ==========================================================================

class BistablePresetBehaviourTest(unittest.TestCase):
    """Bistable preset: two initial conditions must settle to two DIFFERENT states,
    and stay apart when given far longer -- the check that separates a real second
    attractor from a slow transient."""

    MIN_SEPARATION = 0.5          # the preset's own bistability target (min: 0.5)
    LOW, HIGH = 0.0, 10.0

    @classmethod
    def setUpClass(cls):
        cls.t_max = BISTABLE_BLUEPRINT["simulation_config"]["t_max"]
        cls.near = bistable_finals(BISTABLE_BLUEPRINT, BISTABLE_READOUT, cls.t_max,
                                   low=cls.LOW, high=cls.HIGH)
        cls.far = bistable_finals(BISTABLE_BLUEPRINT, BISTABLE_READOUT, 10.0 * cls.t_max,
                                  low=cls.LOW, high=cls.HIGH)

    def test_two_states_separate_at_the_preset_horizon(self):
        """Required: relative separation >= 0.5 (the preset's own bistability target)
        between the branch started at 0 and the branch started at 10. Measured at
        t_max = 100: OFF 0.040005, ON 1.896486, separation 0.9789 (passes)."""
        low, high = self.near
        sep = relative_separation(low, high)
        self.assertGreaterEqual(
            sep, self.MIN_SEPARATION,
            "%s is not bistable at t_max = %g: starting from %.1f settles at %.6g and starting "
            "from %.1f settles at %.6g -- relative separation %.6g (required >= %.2f)."
            % (BISTABLE_READOUT, self.t_max, self.LOW, low, self.HIGH, high,
               sep, self.MIN_SEPARATION))

    def test_separation_survives_ten_times_the_horizon(self):
        """This is the assertion that catches a slow transient masquerading as
        bistability: a short horizon can show a gap that is still closing. Required:
        the separation at 10*t_max is itself >= 0.5 AND has kept at least 20% of the
        gap seen at t_max. Measured at t = 1000: OFF 0.040005, ON 1.896486,
        separation 0.9789 -- identical to the t_max reading (passes)."""
        near_low, near_high = self.near
        far_low, far_high = self.far
        near_sep = relative_separation(near_low, near_high)
        far_sep = relative_separation(far_low, far_high)
        self.assertGreaterEqual(
            far_sep, self.MIN_SEPARATION,
            "the two %s states merge on a longer run: separation %.6g at t = %g (was %.6g at "
            "t_max = %g; required >= %.2f). Finals: low %.6g, high %.6g."
            % (BISTABLE_READOUT, far_sep, 10.0 * self.t_max, near_sep, self.t_max,
               self.MIN_SEPARATION, far_low, far_high))
        self.assertGreaterEqual(
            far_sep, 0.2 * near_sep,
            "the apparent bistability at t_max = %g is a decaying transient: separation fell "
            "from %.6g to %.6g by t = %g (kept %.1f%%, required >= 20%%)."
            % (self.t_max, near_sep, far_sep, 10.0 * self.t_max,
               100.0 * far_sep / max(near_sep, 1e-12)))


# ==========================================================================
# FOLD-CHANGE
# ==========================================================================

class FoldChangePresetBehaviourTest(unittest.TestCase):
    """Fold-change preset: the response to a FIXED fold change of the input must be
    the same at every absolute input level (Weber's law).

    Two levels can never establish this -- a saturating dose-response agrees at two
    nearby levels and falls apart across a span -- so the assay sweeps 5 levels over
    a 100x span.
    """

    LEVELS = [0.3, 1.0, 3.0, 10.0, 30.0]      # 100x span, 5 levels
    TOLERANCE = 0.15                          # 15% agreement
    MIN_RESPONSE = 0.05                       # 5% overshoot counts as "responded"

    @classmethod
    def setUpClass(cls):
        cls.t_max = FOLDCHANGE_BLUEPRINT["simulation_config"]["t_max"]
        cls.fold = float((FOLDCHANGE_BLUEPRINT.get("parameters") or {}).get("fold", 3.0))
        cls.responses, cls.detail = {}, {}
        for level in cls.LEVELS:
            resp, baseline, peak = fold_change_response(
                FOLDCHANGE_BLUEPRINT, FOLDCHANGE_INPUT, FOLDCHANGE_OUTPUT,
                level, t_max=cls.t_max, fold=cls.fold)
            cls.responses[level] = resp
            cls.detail[level] = (baseline, peak)

    def _table(self):
        return "; ".join(
            "%s@%g: baseline %.4g, peak %.4g, response %.4g"
            % (FOLDCHANGE_OUTPUT, lvl, self.detail[lvl][0], self.detail[lvl][1],
               self.responses[lvl])
            for lvl in self.LEVELS)

    def test_the_step_under_test_is_about_threefold(self):
        """The invariance claim is only meaningful for a stated fold change; the brief
        and the preset both use ~3x. Measured: fold = 3.0 (passes)."""
        self.assertTrue(
            2.5 <= self.fold <= 3.5,
            "this assay is written for a ~3x input step but the preset applies %.3gx; "
            "re-state the tolerance for the new step size before trusting the numbers."
            % self.fold)

    def test_every_level_produces_a_response(self):
        """A fold-change detector must respond to the step wherever it sits: required
        >= 5% relative overshoot at every level. Measured: 1.4101, 1.4090, 1.4087,
        1.4085, 1.4085 (passes). This is the assertion a saturating cascade fails --
        the previous Hill version gave 0.00099 at input 30."""
        dead = {lvl: r for lvl, r in self.responses.items() if abs(r) < self.MIN_RESPONSE}
        self.assertEqual(
            {}, dead,
            "the %s -> %s response dies at some input levels: %s (required >= %.0f%% "
            "relative overshoot at EVERY level). Full sweep: %s"
            % (FOLDCHANGE_INPUT, FOLDCHANGE_OUTPUT,
               ", ".join("input %g -> %.4g" % (l, r) for l, r in sorted(dead.items())),
               100 * self.MIN_RESPONSE, self._table()))

    def test_response_is_invariant_across_a_100x_level_span(self):
        """Required: the peak response to the same ~3x step agrees within 15% across 5
        input levels spanning 100x (0.3 -> 30). Measured spread: 0.11% (passes)."""
        values = [self.responses[lvl] for lvl in self.LEVELS]
        span = max(self.LEVELS) / min(self.LEVELS)
        self.assertGreaterEqual(span, 30.0,
                                "the level sweep only spans %.1fx (required >= 30x)" % span)
        biggest = max(abs(v) for v in values)
        self.assertGreater(biggest, self.MIN_RESPONSE,
                           "no level produced any response: %s" % self._table())
        spread = (max(values) - min(values)) / biggest
        self.assertLessEqual(
            spread, self.TOLERANCE,
            "the %s response is not fold-change invariant: responses to the same %.1fx step "
            "differ by %.2f%% across a %.0fx span of absolute input levels (required <= "
            "%.0f%%). %s"
            % (FOLDCHANGE_OUTPUT, self.fold, 100 * spread, span, 100 * self.TOLERANCE,
               self._table()))


# ==========================================================================
# BERRIDGE & GOLDBETER 1990
# ==========================================================================

class BerridgePresetBehaviourTest(unittest.TestCase):
    """Berridge & Goldbeter (1990): sustained cytosolic Ca2+ oscillations (CICR).
    Published hallmark: self-sustained spikes with a regular period."""

    @classmethod
    def setUpClass(cls):
        cls.t_max = BERRIDGE_BLUEPRINT["simulation_config"]["t_max"]
        cls.t, cls.sp = simulate(BERRIDGE_BLUEPRINT, t_max=cls.t_max, num_points=1500)
        cls.y = cls.sp[BERRIDGE_READOUT]
        # A three-times-longer run is what shows the spikes are self-sustained.
        cls.long_t, cls.long_sp = simulate(BERRIDGE_BLUEPRINT, t_max=3.0 * cls.t_max,
                                           num_points=2000)
        cls.long_y = cls.long_sp[BERRIDGE_READOUT]

    def test_calcium_spikes_at_the_published_horizon(self):
        """Required: >= 4 Ca2+ spikes within the preset's own t_max = 10.
        Measured: 13 peaks, range 0.100 .. 1.239 (passes)."""
        peaks = peak_indices(self.y)
        self.assertGreaterEqual(
            len(peaks), 4,
            "cytosolic Ca2+ (%s) spikes only %d times over t_max = %g (required >= 4). Range "
            "%.4g .. %.4g." % (BERRIDGE_READOUT, len(peaks), self.t_max,
                               float(self.y.min()), float(self.y.max())))

    def test_period_is_regular(self):
        """Required: a positive mean period with interval variability <= 0.15.
        Measured: period 0.699 time units, std/mean 0.006 (passes)."""
        times, mean_period, cv = period_stats(self.long_t, self.long_y)
        self.assertGreaterEqual(len(times), 4,
                                "only %d Ca2+ spikes over t = %g; no period to measure"
                                % (len(times), 3.0 * self.t_max))
        self.assertGreater(mean_period, 0.0, "mean period %.4g is not positive" % mean_period)
        self.assertLessEqual(
            cv, 0.15,
            "the Ca2+ period is irregular: mean %.4f, std/mean %.4f (required <= 0.15)"
            % (mean_period, cv))

    def test_oscillation_is_sustained_over_three_horizons(self):
        """Required: amplitude in the final fifth >= 80% of the second fifth over a 3x
        horizon -- self-sustained, not a damped burst. Measured: 0.9584 vs 0.9627,
        ratio 0.9955 (passes)."""
        early = fifth_amplitude(self.long_y, 2)
        late = fifth_amplitude(self.long_y, 5)
        self.assertGreater(early, 1e-6,
                           "Ca2+ is flat in the second fifth (amplitude %.3g)" % early)
        ratio = late / early
        self.assertGreaterEqual(
            ratio, 0.8,
            "the Ca2+ oscillation damps out: amplitude %.4f in the final fifth vs %.4f in the "
            "second fifth (ratio %.4f, required >= 0.80) over t = %g"
            % (late, early, ratio, 3.0 * self.t_max))


# ==========================================================================
# ZHABOTINSKY 2000
# ==========================================================================

class ZhabotinskyPresetBehaviourTest(unittest.TestCase):
    """Zhabotinsky (2000): a TRANSIENT Ca2+ pulse latches CaMKII permanently ON.

    The published claim has two halves and both are asserted: the pulse must flip the
    switch and it must stay flipped after the pulse is gone, AND the OFF state must be
    stable at the same baseline Ca2+ -- otherwise the kinase is merely Ca2+-driven,
    not bistable.
    """

    ON_LEVEL = 10.0          # active CaMKII considered "ON" (paper/preset value ~15)
    OFF_LEVEL = 1.0          # active CaMKII considered "OFF"

    @classmethod
    def setUpClass(cls):
        cls.t_max = ZHABOTINSKY_BLUEPRINT["simulation_config"]["t_max"]
        cls.t, cls.sp = simulate(ZHABOTINSKY_BLUEPRINT, t_max=cls.t_max, num_points=2000)
        cls.y = cls.sp[ZHABOTINSKY_READOUT]
        cls.long_t, cls.long_sp = simulate(ZHABOTINSKY_BLUEPRINT, t_max=3.0 * cls.t_max,
                                           num_points=2000)
        cls.long_y = cls.long_sp[ZHABOTINSKY_READOUT]
        # Control: identical model, pulse amplitude zeroed.
        cls.ctrl_t, cls.ctrl_sp = simulate(ZHABOTINSKY_BLUEPRINT, t_max=cls.t_max,
                                           num_points=2000, custom_params={"amp": 0.0})
        cls.ctrl_y = cls.ctrl_sp[ZHABOTINSKY_READOUT]

    def test_transient_pulse_latches_the_switch_on(self):
        """Required: active CaMKII is OFF (< 1.0) before the pulse at t = 20 and ON
        (>= 10.0) well after the pulse ends at t = 60. Measured: 0.193 at t = 15,
        14.15 at t = 70, 15.42 at t = 350 (passes)."""
        before = value_at(self.t, self.y, ZHAB_PULSE_ON - 5.0)
        after = value_at(self.t, self.y, ZHAB_PULSE_OFF + 10.0)
        final = float(self.y[-1])
        self.assertLessEqual(
            before, self.OFF_LEVEL,
            "active CaMKII is not OFF before the pulse: %s = %.4f at t = %g (required <= %.1f)"
            % (ZHABOTINSKY_READOUT, before, ZHAB_PULSE_ON - 5.0, self.OFF_LEVEL))
        self.assertGreaterEqual(
            after, self.ON_LEVEL,
            "the Ca2+ pulse did not switch CaMKII on: %s = %.4f at t = %g, after the pulse "
            "ended at t = %g (required >= %.1f)"
            % (ZHABOTINSKY_READOUT, after, ZHAB_PULSE_OFF + 10.0, ZHAB_PULSE_OFF,
               self.ON_LEVEL))
        self.assertGreaterEqual(
            final, self.ON_LEVEL,
            "CaMKII did not stay on to the end of the shipped run: %s = %.4f at t_max = %g "
            "(required >= %.1f)"
            % (ZHABOTINSKY_READOUT, final, self.t_max, self.ON_LEVEL))

    def test_latched_state_persists_at_three_times_the_horizon(self):
        """'Permanently ON' means the latch outlives the plotted window. Required: at
        3*t_max the level is still >= 10.0 and >= 80% of its value at t_max. Measured:
        15.42 at t = 350 and 15.57 at t = 1050 (passes)."""
        at_tmax = float(self.y[-1])
        at_long = float(self.long_y[-1])
        self.assertGreaterEqual(
            at_long, self.ON_LEVEL,
            "the CaMKII latch decays on a longer run: %s = %.4f at t = %g (required >= %.1f; "
            "it was %.4f at t_max = %g)"
            % (ZHABOTINSKY_READOUT, at_long, 3.0 * self.t_max, self.ON_LEVEL,
               at_tmax, self.t_max))
        self.assertGreaterEqual(
            at_long, 0.8 * at_tmax,
            "the latched state is drifting down: %.4f at t = %g vs %.4f at t_max = %g (kept "
            "%.1f%%, required >= 80%%)"
            % (at_long, 3.0 * self.t_max, at_tmax, self.t_max,
               100.0 * at_long / max(at_tmax, 1e-12)))

    def test_off_state_is_stable_without_a_pulse(self):
        """Without the pulse (amp = 0) the same baseline Ca2+ must leave CaMKII OFF --
        that is what makes the switch bistable rather than Ca2+-driven. Required:
        max < 1.0 over the whole run. Measured: 0.243 (passes)."""
        peak = float(self.ctrl_y.max())
        self.assertLess(
            peak, self.OFF_LEVEL,
            "CaMKII switches on WITHOUT a pulse (amp = 0): %s reaches %.4f (required < %.1f). "
            "The OFF state is not stable at baseline Ca2+ = %s, so the 'memory' is just the "
            "stimulus." % (ZHABOTINSKY_READOUT, peak, self.OFF_LEVEL,
                           ZHABOTINSKY_BLUEPRINT["parameters"]["Cabase"]))


# ==========================================================================
# LYASHENKO 2020
# ==========================================================================

class LyashenkoPresetBehaviourTest(unittest.TestCase):
    """Lyashenko et al. (2020): fold-change detection. Equal fold steps of ligand give
    equal response pulses and the response re-adapts to baseline between steps -- at
    ANY absolute ligand level (Weber's law)."""

    TOLERANCE = 0.15          # 15% spread across pulses / across absolute levels
    SCALE = 30.0              # absolute-level shift used for the invariance test

    @classmethod
    def setUpClass(cls):
        cls.t_max = LYASHENKO_BLUEPRINT["simulation_config"]["t_max"]
        cls.t, cls.sp = simulate(LYASHENKO_BLUEPRINT, t_max=cls.t_max, num_points=2000)
        cls.y = cls.sp[LYASHENKO_READOUT]
        cls.heights = [float(cls.y[i]) for i in peak_indices(cls.y)]
        # Same 2x staircase, every absolute level multiplied by SCALE.
        scaled_params = {"L%d" % (i + 1): level * cls.SCALE
                         for i, level in enumerate(LYASHENKO_LEVELS)}
        cls.scaled_t, cls.scaled_sp = simulate(
            LYASHENKO_BLUEPRINT, t_max=cls.t_max, num_points=2000,
            custom_params=scaled_params,
            custom_initial={"L": LYASHENKO_LEVELS[0] * cls.SCALE,
                            "R": LYASHENKO_LEVELS[0] * cls.SCALE,
                            "S": LYASHENKO_BASELINE})
        cls.scaled_y = cls.scaled_sp[LYASHENKO_READOUT]
        cls.scaled_heights = [float(cls.scaled_y[i]) for i in peak_indices(cls.scaled_y)]

    def test_equal_fold_steps_give_equal_response_pulses(self):
        """The shipped staircase is four 2x steps (1->2->4->8->16). Required: four
        pulses whose heights agree within 15%. Measured: 1.8155, 1.8215, 1.8218,
        1.8218 -- spread 0.34% (passes)."""
        self.assertEqual(
            len(LYASHENKO_LEVELS) - 1, len(self.heights),
            "expected %d response pulses in %s (one per 2x ligand step) but found %d: heights "
            "%s" % (len(LYASHENKO_LEVELS) - 1, LYASHENKO_READOUT, len(self.heights),
                    [round(h, 4) for h in self.heights]))
        spread = (max(self.heights) - min(self.heights)) / max(self.heights)
        self.assertLessEqual(
            spread, self.TOLERANCE,
            "the fold-change response is not uniform across equal fold steps: peak heights %s "
            "spread by %.2f%% (required <= %.0f%%)"
            % ([round(h, 4) for h in self.heights], 100 * spread, 100 * self.TOLERANCE))

    def test_response_re_adapts_to_baseline_after_each_step(self):
        """FCD requires the readout to return to baseline between steps, or the 'pulse
        height' is just a rising level. Required: final S within 10% of the baseline
        1.0. Measured: 1.0013 (passes)."""
        final = float(self.y[-1])
        drift = abs(final - LYASHENKO_BASELINE) / LYASHENKO_BASELINE
        self.assertLessEqual(
            drift, 0.10,
            "%s does not re-adapt: final value %.4f vs baseline %.4f (off by %.1f%%, required "
            "<= 10%%)" % (LYASHENKO_READOUT, final, LYASHENKO_BASELINE, 100 * drift))

    def test_response_is_invariant_to_absolute_ligand_level(self):
        """The Weber's-law claim: multiply every absolute level by 30 and the same 2x
        steps must give the same pulse heights. Required: agreement within 15%.
        Measured: identical to 4 decimals (1.8155, 1.8215, 1.8218, 1.8218) (passes)."""
        self.assertEqual(
            len(self.heights), len(self.scaled_heights),
            "shifting every ligand level by %gx changed the number of response pulses: %d at "
            "1x vs %d at %gx (heights %s vs %s)"
            % (self.SCALE, len(self.heights), len(self.scaled_heights), self.SCALE,
               [round(h, 4) for h in self.heights],
               [round(h, 4) for h in self.scaled_heights]))
        worst = max(abs(a - b) / max(abs(a), abs(b), 1e-12)
                    for a, b in zip(self.heights, self.scaled_heights))
        self.assertLessEqual(
            worst, self.TOLERANCE,
            "the response depends on the ABSOLUTE ligand level, so this is not fold-change "
            "detection: pulse heights %s at 1x vs %s at %gx -- worst mismatch %.2f%% "
            "(required <= %.0f%%)"
            % ([round(h, 4) for h in self.heights],
               [round(h, 4) for h in self.scaled_heights], self.SCALE,
               100 * worst, 100 * self.TOLERANCE))


# ==========================================================================
# DRIFT GUARDS  (are the frozen copies above still the shipped presets?)
# ==========================================================================

class PresetDefinitionDriftTest(unittest.TestCase):
    """The definitions at the top of this file are frozen copies of what
    static/app.js ships. These tests fail when the shipped definition moves, so a
    behavioural result is never silently measured against a stale model.

    They never build a blueprint out of JavaScript: one searches app.js for the frozen
    TEXT, one searches it for the frozen EQUATIONS and PARAMETER VALUES, and one
    re-derives the text-only Turing preset through the same deterministic parser the
    backend uses.
    """

    # Turing is the only preset that still ships text only.
    PARSED_PRESETS = {"turing": (TURING_TEXT, TURING_BLUEPRINT)}
    @staticmethod
    def _normalise(text):
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def setUpClass(cls):
        cls.app_js = None
        if os.path.exists(APP_JS):
            with open(APP_JS, "r", encoding="utf-8") as handle:
                cls.app_js = cls._normalise(handle.read())

    def _require_app_js(self):
        if self.app_js is None:
            self.skipTest("static/app.js not found at %s" % APP_JS)

    def test_frozen_preset_texts_are_still_the_shipped_texts(self):
        self._require_app_js()
        missing = [name for name, text in sorted(FROZEN_TEXTS.items())
                   if self._normalise(text) not in self.app_js]
        self.assertEqual(
            [], missing,
            "the preset text frozen in this file no longer appears in static/app.js for: %s. "
            "The presets were edited; refresh the frozen text AND blueprint in "
            "test_preset_behaviour.py so these behaviour tests measure the shipped model."
            % ", ".join(missing))

    def test_frozen_equations_and_parameters_are_still_the_shipped_ones(self):
        """Every ODE right-hand side, shared flux and parameter value frozen above must
        still be present in app.js. This is what catches a retune that changes the
        model without changing the preset's text."""
        self._require_app_js()
        problems = []
        for name, blueprint in sorted(FROZEN_BLUEPRINTS.items()):
            if name in self.PARSED_PRESETS:
                # Its equations come from agent.rule_based_parse, not app.js;
                # test_text_only_presets_still_compile_to_the_frozen_blueprints covers it.
                continue
            for kind in ("odes", "fluxes"):
                for key, expression in (blueprint.get(kind) or {}).items():
                    if self._normalise(expression) not in self.app_js:
                        problems.append("%s: %s[%s] = %r is not in app.js"
                                        % (name, kind, key, expression))
            for pname, pvalue in (blueprint.get("parameters") or {}).items():
                value = float(pvalue)
                candidates = {"%s: %r" % (pname, value), "%s: %g" % (pname, value)}
                if value.is_integer():
                    candidates.add("%s: %d" % (pname, int(value)))
                if not any(candidate in self.app_js for candidate in candidates):
                    problems.append("%s: parameter %s = %r is not in app.js"
                                    % (name, pname, value))
            spatial = blueprint.get("spatial") or {}
            for key, expression in (spatial.get("reactions") or {}).items():
                if self._normalise(expression) not in self.app_js:
                    problems.append("%s: reaction[%s] = %r is not in app.js"
                                    % (name, key, expression))
        self.assertEqual(
            [], problems,
            "the frozen preset equations/parameters no longer match static/app.js:\n  "
            + "\n  ".join(problems)
            + "\nRefresh the frozen blueprints in test_preset_behaviour.py, then re-read the "
              "behavioural results -- they were measured on the OLD model.")

    def test_text_only_presets_still_compile_to_the_frozen_blueprints(self):
        problems = []
        for name, (text, frozen) in sorted(self.PARSED_PRESETS.items()):
            parsed = agent.rule_based_parse(text)
            if parsed.get("type") != frozen["type"]:
                problems.append("%s: type %s != %s" % (name, parsed.get("type"), frozen["type"]))
                continue
            got_nodes = sorted((n["id"], float(n.get("initial_value", 0.0)))
                               for n in parsed.get("nodes", []))
            want_nodes = sorted((n["id"], float(n.get("initial_value", 0.0)))
                                for n in frozen["nodes"])
            if got_nodes != want_nodes:
                problems.append("%s: nodes %s != %s" % (name, got_nodes, want_nodes))
            got_edges = sorted((e["source"], e["target"], e.get("type", "activation"),
                                tuple(sorted((e.get("parameters") or {}).items())))
                               for e in parsed.get("edges", []))
            want_edges = sorted((e["source"], e["target"], e.get("type", "activation"),
                                 tuple(sorted((e.get("parameters") or {}).items())))
                                for e in frozen["edges"])
            if got_edges != want_edges:
                problems.append("%s: edges %s != %s" % (name, got_edges, want_edges))
            if frozen["type"] == "PDE":
                got_spatial = parsed.get("spatial", {})
                for key in ("diffusion", "reactions"):
                    if got_spatial.get(key) != frozen["spatial"][key]:
                        problems.append("%s: spatial.%s %s != %s"
                                        % (name, key, got_spatial.get(key),
                                           frozen["spatial"][key]))
                got_tmax = float(parsed.get("simulation_config", {}).get("t_max", 0.0))
                frozen_tmax = float(frozen["simulation_config"]["t_max"])
                if got_tmax != frozen_tmax:
                    problems.append("%s: t_max %.3f != %.3f" % (name, got_tmax, frozen_tmax))
        self.assertEqual(
            [], problems,
            "agent.rule_based_parse no longer produces the blueprint frozen in this file:\n  "
            + "\n  ".join(problems)
            + "\nRefresh the frozen blueprint so the behaviour tests measure the shipped model.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
