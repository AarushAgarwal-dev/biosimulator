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
    PASS EGF/EGFR    : ERK peaks at t = 6.68 with value 0.6074 and falls to 46% of
                       that peak by t_max = 50; EGFR decays to 0.66% of its peak.
                       (This preset is why the frozen copies are gone: it was fixed
                       in app.js while this file still measured the old model.)
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
They are LOADED, never copied. ``preset_loader`` reads the ``Presets`` and
``PaperModels`` object literals out of ``static/app.js`` at import time -- a small
tolerant reader, no JS engine, no ``eval``, standard library only -- and returns
them as plain dicts:

  * egfr, oscillator, bistable, foldchange (Presets) and berridge, zhabotinsky,
    lyashenko (PaperModels) ship an explicit ``blueprint`` object: used as loaded.
  * turing ships only ``text`` + ``targets``; the blueprint the user actually runs
    is what ``agent.rule_based_parse(text)`` returns (the deterministic, LLM-off
    path behind ``POST /api/blueprint``), so the loader compiles it through exactly
    that function.
  * each preset's readout species is taken from the preset's OWN target list rather
    than named here, so a retargeted preset retargets its behavioural test with it.

This replaced a frozen Python transcription of all eight presets, which drifted
exactly as you would expect: the EGF/EGFR preset was fixed in app.js and this file
went on asserting against the stale copy until the two were hand-synced. There is
now no copy to sync -- retune a preset in app.js and it is measured in its retuned
form, in the same commit.

Three guards keep the loading honest:

  * ``PresetLoaderTest`` -- the loader finds all eight presets, each with nodes and
    edges, and MALFORMED INPUT RAISES. That last one is the important one: a loader
    that answered ``{}`` when confused would make every behavioural test below pass
    while measuring nothing, which is worse than the drift it replaces.
  * ``ShippedDefinitionFidelityTest`` -- every text, equation and parameter value
    that was actually measured still appears verbatim in app.js, so a reader bug
    cannot masquerade as a green run. (It only searches app.js for strings; it never
    builds a blueprint out of JavaScript.)
  * ``PresetCoverageTest`` -- every preset app.js ships has a behavioural test
    class, so a ninth preset cannot arrive untested.

Run just this tier while iterating:

    python -m unittest test_preset_behaviour
"""

import os
import re
import unittest

import numpy as np
from scipy.signal import find_peaks

import agent
import preset_loader
from simulation_engine import ODEModel, solve_pde

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
APP_JS = os.path.join(REPO_DIR, "static", "app.js")

# Deterministic noise for the PDE preset (solve_pde seeds its grids from numpy's
# global RNG, so an unseeded run would make the Turing assertions flaky).
PDE_SEED = 20260913


# ==========================================================================
# THE SHIPPED PRESET DEFINITIONS -- LOADED FROM static/app.js, NOT COPIED
# ==========================================================================
# static/app.js is the only place a preset is defined, and preset_loader reads the
# `Presets` and `PaperModels` object literals straight out of it at import time.
# Nothing below is a transcription: retune a preset in app.js and these tests
# measure the retuned model in the same commit, with no hand-sync step.
#
# That hand-sync step is what failed before. The EGF/EGFR preset was fixed in
# app.js while this file kept a stale Python copy, so its test went on failing
# against the old model -- a red test naming a preset that was already correct and,
# in the other direction, a green test that proves nothing about what ships.
#
# preset_loader RAISES rather than returning anything partial (PresetLoaderTest
# below pins that down): a silently empty preset table would make every assertion
# in this module pass while measuring nothing at all.

PRESETS = preset_loader.load_presets()

# turing is the one preset that ships text + targets and NO blueprint -- what the
# user runs is whatever agent.rule_based_parse (the deterministic, LLM-off path
# behind POST /api/blueprint) makes of that text, so the loader compiles it with
# exactly that function. Every other preset ships an explicit blueprint.
BLUEPRINTS = preset_loader.load_blueprints(text_compiler=agent.rule_based_parse)


def _target_species(preset, kind, field="species"):
    """The species a preset's own `kind` target names.

    The readout under test is never chosen by this file -- it is whatever species
    the preset's own target refers to, so a retargeted preset retargets its
    behavioural test with it. Raises if that target is gone, because silently
    re-pointing a behavioural test at another species is how a test starts
    measuring something nobody asked about.
    """
    for target in PRESETS[preset].get("targets") or []:
        if target.get("type") == kind and target.get(field):
            return target[field]
    raise AssertionError(
        "preset %r ships no %r target with a %r, so this module cannot tell which "
        "species its behavioural test should measure. Its app.js targets are: %r"
        % (preset, kind, field, PRESETS[preset].get("targets")))


# --- Turing: the only text-only preset; the blueprint under test is what
# --- agent.rule_based_parse() deterministically compiles the shipped text to ----

TURING_TEXT = PRESETS["turing"]["text"]
TURING_BLUEPRINT = BLUEPRINTS["turing"]

# --- EGF/EGFR ----------------------------------------------------------------

EGFR_BLUEPRINT = BLUEPRINTS["egfr"]
EGFR_TEXT = PRESETS["egfr"]["text"]
# The preset's OWN targets, as shipped. The readout under test is the species its
# peak_time target names, i.e. ERK.
EGFR_TARGETS = PRESETS["egfr"]["targets"]
EGFR_READOUT = _target_species("egfr", "peak_time")

# --- Oscillator: Goodwin (1965) three-stage negative feedback, Hill n = 16 ----

OSCILLATOR_BLUEPRINT = BLUEPRINTS["oscillator"]
OSCILLATOR_TEXT = PRESETS["oscillator"]["text"]
OSCILLATOR_READOUT = _target_species("oscillator", "oscillation")

# --- Bistable: cooperative positive autofeedback + first-order removal --------

BISTABLE_BLUEPRINT = BLUEPRINTS["bistable"]
BISTABLE_TEXT = PRESETS["bistable"]["text"]
BISTABLE_READOUT = _target_species("bistable", "bistability")

# --- Fold-change: incoherent feed-forward loop (Goentoro & Alon 2009) ---------

FOLDCHANGE_BLUEPRINT = BLUEPRINTS["foldchange"]
FOLDCHANGE_TEXT = PRESETS["foldchange"]["text"]
# Which species is the input and which the response comes from the preset's own
# fold_change target, not from a choice made in this file.
FOLDCHANGE_INPUT = _target_species("foldchange", "fold_change", field="input")
FOLDCHANGE_OUTPUT = _target_species("foldchange", "fold_change", field="output")

# --- Berridge & Goldbeter (1990) ---------------------------------------------

BERRIDGE_BLUEPRINT = BLUEPRINTS["berridge"]
BERRIDGE_READOUT = _target_species("berridge", "oscillation")        # cytosolic Ca2+

# --- Zhabotinsky (2000) ------------------------------------------------------

ZHABOTINSKY_BLUEPRINT = BLUEPRINTS["zhabotinsky"]
ZHABOTINSKY_READOUT = _target_species("zhabotinsky", "steady_state")  # active CaMKII
# The pulse window is the model's own t_on/t_off, so retiming the Ca2+ pulse in
# app.js retimes the before/after sampling here with it.
ZHAB_PULSE_ON = float(ZHABOTINSKY_BLUEPRINT["parameters"]["t_on"])
ZHAB_PULSE_OFF = float(ZHABOTINSKY_BLUEPRINT["parameters"]["t_off"])

# --- Lyashenko et al. (2020) -------------------------------------------------

LYASHENKO_BLUEPRINT = BLUEPRINTS["lyashenko"]
LYASHENKO_READOUT = _target_species("lyashenko", "steady_state")
# The shipped ligand staircase, read off the model's own L1, L2, ... parameters
# (currently a 2x ladder over a full decade), so adding a step to app.js adds it
# to the assertions instead of leaving this list short.
_LYASHENKO_LEVEL_PARAMS = sorted(
    (int(pname[1:]), pvalue)
    for pname, pvalue in LYASHENKO_BLUEPRINT["parameters"].items()
    if re.fullmatch(r"L\d+", pname))
LYASHENKO_LEVELS = [float(pvalue) for _index, pvalue in _LYASHENKO_LEVEL_PARAMS]
if len(LYASHENKO_LEVELS) < 2:
    raise AssertionError(
        "the lyashenko preset ships %d ligand-level parameters (L1, L2, ...); "
        "fold-change detection needs at least two steps to compare"
        % len(LYASHENKO_LEVELS))
LYASHENKO_BASELINE = 1.0          # S tracks the ratio L/R, so it re-adapts to 1

# Every preset as loaded, for the fidelity guard and the coverage guard.
LOADED_BLUEPRINTS = dict(BLUEPRINTS)
LOADED_TEXTS = {name: record["text"] for name, record in PRESETS.items()
                if (record.get("text") or "").strip()}


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

class ShippedDefinitionFidelityTest(unittest.TestCase):
    """The definitions the behavioural tests above ran on were LOADED out of
    static/app.js by preset_loader, not copied into this file. These tests check the
    loading was FAITHFUL: every text, equation and parameter value that was measured
    must still be findable verbatim in the app.js source. That is what catches a
    reader bug -- a mangled expression, a mis-parsed number, an escape decoded
    wrongly -- which would otherwise look like a perfectly green behavioural run
    against a model nobody ships.

    They never evaluate JavaScript: two of them search the raw app.js text for the
    strings the loader produced, and one re-derives the text-only Turing preset
    through the same deterministic parser the backend uses.
    """

    # Presets that ship text only: their equations come from agent.rule_based_parse,
    # so those are not in app.js to be found. Derived, so a preset that gains an
    # explicit blueprint moves into the equation check on its own.
    PARSED_PRESETS = tuple(sorted(name for name, record in PRESETS.items()
                                  if "blueprint" not in record))

    @staticmethod
    def _normalise(text):
        return re.sub(r"\s+", " ", text).strip()

    def _appears_quoted(self, expression):
        """Is ``expression`` in app.js as a whole double-quoted string?

        Bounded by the quotes app.js writes every rate law inside, because an
        unbounded substring search cannot tell a faithful read from a TRUNCATED one:
        a prefix of a shipped expression is still a substring of the file.
        """
        return '"%s"' % self._normalise(expression) in self.app_js

    def _appears_as_parameter(self, pname, value):
        """Is ``pname: value`` in app.js, with the delimiter that ends the value?

        The delimiter matters for the same reason: without it `d1: 0.15` matches a
        shipped `d1: 0.1500001`. Spellings are round-trip exact (`repr`, plus the
        integer form for whole numbers) -- deliberately not `%g`, whose 6 significant
        digits would round a mutated value back onto the shipped one.
        """
        spellings = {repr(value)}
        if value.is_integer():
            spellings.add(str(int(value)))
        return any("%s: %s%s" % (pname, spelling, tail) in self.app_js
                   for spelling in spellings for tail in (",", " ", "}"))

    @classmethod
    def setUpClass(cls):
        cls.app_js = None
        if os.path.exists(APP_JS):
            with open(APP_JS, "r", encoding="utf-8") as handle:
                cls.app_js = cls._normalise(handle.read())

    def _require_app_js(self):
        if self.app_js is None:
            self.skipTest("static/app.js not found at %s" % APP_JS)

    def test_loaded_preset_texts_are_the_shipped_texts(self):
        self._require_app_js()
        self.assertTrue(LOADED_TEXTS,
                        "no preset text was loaded at all -- preset_loader returned "
                        "presets with no `text`, so the text-driven presets are being "
                        "tested against nothing")
        missing = [name for name, text in sorted(LOADED_TEXTS.items())
                   if self._normalise(text) not in self.app_js]
        self.assertEqual(
            [], missing,
            "preset_loader produced text for %s that does not appear in static/app.js. "
            "The loader mangled the template literal it read (escapes? line "
            "continuations?), so these presets were parsed from something app.js does "
            "not contain." % ", ".join(missing))

    def test_loaded_equations_and_parameters_are_the_shipped_ones(self):
        """Every ODE right-hand side, shared flux and parameter value the behavioural
        tests integrated must still be present in app.js. This is the check that a
        retune reaches the tests -- and that the reader reproduced it exactly."""
        self._require_app_js()
        problems = []
        for name, blueprint in sorted(LOADED_BLUEPRINTS.items()):
            if name in self.PARSED_PRESETS:
                # Compiled from text by agent.rule_based_parse, not shipped as
                # equations; test_text_only_presets_compile_to_the_model_under_test
                # covers it.
                continue
            for kind in ("odes", "fluxes"):
                for key, expression in (blueprint.get(kind) or {}).items():
                    if not self._appears_quoted(expression):
                        problems.append("%s: %s[%s] = %r is not in app.js"
                                        % (name, kind, key, expression))
            for pname, pvalue in (blueprint.get("parameters") or {}).items():
                if isinstance(pvalue, bool) or not isinstance(pvalue, (int, float)):
                    problems.append("%s: parameter %s = %r is not a number"
                                    % (name, pname, pvalue))
                    continue
                if not self._appears_as_parameter(pname, float(pvalue)):
                    problems.append("%s: parameter %s = %r is not in app.js"
                                    % (name, pname, float(pvalue)))
            spatial = blueprint.get("spatial") or {}
            for key, expression in (spatial.get("reactions") or {}).items():
                if not self._appears_quoted(expression):
                    problems.append("%s: reaction[%s] = %r is not in app.js"
                                    % (name, key, expression))
        self.assertEqual(
            [], problems,
            "preset_loader produced equations/parameters that are not in static/app.js:"
            "\n  " + "\n  ".join(problems)
            + "\nThe behavioural results above were measured on a model app.js does "
              "not define, which means the reader in preset_loader.py is wrong -- fix "
              "it there, do not hand-write the value here.")

    def test_text_only_presets_compile_to_the_model_under_test(self):
        """A text-only preset has no shipped equations, so what it is tested on is
        whatever the deterministic parser makes of its text. Assert that parse is
        stable and that it honours the species the text itself declares."""
        problems = []
        for name in self.PARSED_PRESETS:
            text = PRESETS[name]["text"]
            blueprint = LOADED_BLUEPRINTS[name]
            if agent.rule_based_parse(text) != blueprint:
                problems.append("%s: agent.rule_based_parse(text) is not stable -- two "
                                "calls on the same shipped text disagree, so the "
                                "behaviour tested is not reproducible" % name)
            declared = {species: float(value) for species, value
                        in re.findall(r"^(\w+) starts at ([0-9.]+)\.", text, re.M)}
            got = {node["id"]: float(node.get("initial_value", 0.0))
                   for node in blueprint.get("nodes") or []}
            if not declared:
                problems.append("%s: the shipped text declares no `X starts at V.` "
                                "line, so nothing pins its initial state" % name)
            elif got != declared:
                problems.append("%s: compiled nodes %s do not match the initial values "
                                "the shipped text declares, %s" % (name, got, declared))
            if blueprint.get("type") == "PDE":
                spatial = blueprint.get("spatial") or {}
                for key in ("diffusion", "reactions"):
                    absent = sorted(s for s in got if s not in (spatial.get(key) or {}))
                    if absent:
                        problems.append("%s: PDE blueprint has no spatial.%s for %s"
                                        % (name, key, ", ".join(absent)))
                t_max = float((blueprint.get("simulation_config") or {}).get("t_max", 0.0))
                if t_max <= 0.0:
                    problems.append("%s: PDE blueprint has t_max %.3f" % (name, t_max))
        self.assertEqual(
            [], problems,
            "the text-only presets no longer compile to a model these tests can "
            "measure:\n  " + "\n  ".join(problems))


# ==========================================================================
# THE LOADER ITSELF
# ==========================================================================

class PresetLoaderTest(unittest.TestCase):
    """preset_loader is load-bearing for every assertion in this module, so it gets
    its own tests.

    The ones that matter most are the failure tests. A loader that answered ``{}``
    when it could not find or parse a preset would make every behavioural test above
    pass while measuring nothing at all -- strictly worse than the drift it replaces,
    because a stale copy at least still asserts something. So malformed input must
    RAISE, and these tests hold it to that.
    """

    #: Which table each shipped preset must be found in.
    EXPECTED_TABLES = {
        "egfr": "Presets", "turing": "Presets", "oscillator": "Presets",
        "bistable": "Presets", "foldchange": "Presets",
        "berridge": "PaperModels", "zhabotinsky": "PaperModels",
        "lyashenko": "PaperModels",
    }

    # A stand-in for app.js exercising every bit of the dialect the reader must cope
    # with: identifier keys, a template literal spanning lines, // and /* */ comments
    # containing braces and quotes, trailing commas, both number kinds, an escape,
    # true/false/null, and function values that must be skipped rather than refused.
    SAMPLE = """
// leading comment { with a brace } and a "quote"
const Presets = {
    alpha: {
        text: `first line
second line`,
        targets: [
            { species: "X", type: "peak_time", min: 5.0, max: 15.0 },   // trailing ->
        ],
        t_max: 140.0,
        render: (v) => v * 2,
        helper: function (a) { return { a: a }; },
        blueprint: {
            type: "ODE",
            /* a block comment with a brace } a "quote" and a stray colon : */
            nodes: [{ id: "A", initial_value: 1.0 }, { id: "B", initial_value: 0 }],
            edges: [{ source: "A", target: "B", type: "activation" },],
            parameters: { k: 0.5, n: 4, big: 1.5e3, neg: -2.5,
                          on: true, off: false, gone: null },
            odes: { A: "-k*A", B: "k*A - n*B" },
            simulation_config: { t_max: 50.0 },
        },
    },
};
const PaperModels = {
    beta: {
        title: "Beta \\u00b5 model",
        blueprint: {
            nodes: [{ id: "Z" }],
            edges: [{ source: "Z", target: "Z", type: "activation" }],
        },
        targets: [{ species: "Z", type: "oscillation", min: 0.2 }],
    },
};
"""

    #: Every one of these must raise, not return a partial or empty result.
    MALFORMED_LITERALS = (
        ("truncated object", "{ nodes: [ { id: 'A' } "),
        ("unterminated double-quoted string", '{ text: "oops }'),
        ("unterminated template literal", "{ text: `oops }"),
        ("unterminated block comment", "{ /* never closed "),
        ("key with no colon", "{ nodes [ ] }"),
        ("colon with no value", "{ nodes: }"),
        ("bare identifier as a value", "{ blueprint: SOME_OTHER_CONST }"),
        ("expression as a value", "{ t_max: (1 + 2) }"),
        ("array where an object belongs", "[1, 2, 3]"),
        ("empty input", ""),
    )

    PAPER_MODEL_NAMES = ("berridge", "zhabotinsky", "lyashenko")

    @classmethod
    def setUpClass(cls):
        cls.source = preset_loader.read_app_js()

    @classmethod
    def _synthetic_source(cls, **overrides):
        """A stand-in app.js defining every REQUIRED_PRESETS name, so a test can
        make ONE preset defective and see that defect reported."""
        default = ('{ blueprint: { nodes: [{ id: "A", initial_value: 1.0 }], '
                   'edges: [{ source: "A", target: "A", type: "activation" }] }, '
                   'targets: [] }')
        presets, papers = [], []
        for name in preset_loader.REQUIRED_PRESETS:
            entry = "    %s: %s," % (name, overrides.get(name, default))
            (papers if name in cls.PAPER_MODEL_NAMES else presets).append(entry)
        return ("const Presets = {\n%s\n};\n\nconst PaperModels = {\n%s\n};\n"
                % ("\n".join(presets), "\n".join(papers)))

    # -- what it finds --------------------------------------------------------
    def test_finds_all_eight_shipped_presets_by_name(self):
        self.assertEqual(sorted(self.EXPECTED_TABLES),
                         sorted(preset_loader.REQUIRED_PRESETS),
                         "this test and preset_loader disagree about which presets ship")
        for name, table in sorted(self.EXPECTED_TABLES.items()):
            self.assertIn(name, PRESETS,
                          "preset_loader did not find %r in static/app.js; it found %s"
                          % (name, ", ".join(sorted(PRESETS))))
            self.assertEqual(table, PRESETS[name]["table"],
                             "preset %r came from %s, expected the %s table"
                             % (name, PRESETS[name]["table"], table))
        self.assertGreaterEqual(len(PRESETS), 8)

    def test_every_preset_has_nodes_and_edges(self):
        for name in sorted(PRESETS):
            blueprint = BLUEPRINTS[name]
            nodes, edges = blueprint.get("nodes"), blueprint.get("edges")
            self.assertTrue(nodes, "preset %r loaded with no nodes" % name)
            self.assertTrue(edges, "preset %r loaded with no edges" % name)
            for node in nodes:
                self.assertTrue(node.get("id"),
                                "preset %r has a node with no id: %r" % (name, node))
            for edge in edges:
                self.assertTrue(edge.get("source") and edge.get("target"),
                                "preset %r has an edge with no source/target: %r"
                                % (name, edge))

    def test_every_preset_carries_its_own_targets(self):
        for name in sorted(PRESETS):
            targets = PRESETS[name].get("targets")
            self.assertTrue(targets, "preset %r loaded with no targets, so its "
                                     "behavioural readout cannot be derived" % name)
            for target in targets:
                self.assertTrue(target.get("type"),
                                "preset %r has a target with no type: %r" % (name, target))

    def test_reads_the_javascript_dialect_app_js_actually_uses(self):
        tables = preset_loader.load_tables(source=self.SAMPLE)
        self.assertEqual(["Presets", "PaperModels"], list(tables))
        alpha = tables["Presets"]["alpha"]
        self.assertEqual("first line\nsecond line", alpha["text"],
                         "a multi-line template literal was not read verbatim")
        self.assertEqual([{"species": "X", "type": "peak_time",
                           "min": 5.0, "max": 15.0}], alpha["targets"])
        self.assertEqual(140.0, alpha["t_max"])
        blueprint = alpha["blueprint"]
        self.assertEqual([{"id": "A", "initial_value": 1.0},
                          {"id": "B", "initial_value": 0}], blueprint["nodes"])
        self.assertEqual([{"source": "A", "target": "B", "type": "activation"}],
                         blueprint["edges"], "a trailing comma broke the array")
        self.assertEqual({"k": 0.5, "n": 4, "big": 1500.0, "neg": -2.5,
                          "on": True, "off": False, "gone": None},
                         blueprint["parameters"])
        self.assertEqual({"A": "-k*A", "B": "k*A - n*B"}, blueprint["odes"])
        self.assertEqual({"t_max": 50.0}, blueprint["simulation_config"])
        self.assertEqual("Beta \u00b5 model", tables["PaperModels"]["beta"]["title"],
                         r"a \u escape was not decoded")

    def test_skips_function_values_and_keeps_everything_else(self):
        alpha = preset_loader.load_tables(source=self.SAMPLE)["Presets"]["alpha"]
        self.assertNotIn("render", alpha, "an arrow function should be skipped, not kept")
        self.assertNotIn("helper", alpha,
                         "a function expression should be skipped, not kept")
        self.assertEqual(["blueprint", "t_max", "targets", "text"], sorted(alpha),
                         "skipping the function values dropped something else too")

    def test_loaded_blueprints_are_independent_copies(self):
        first = preset_loader.load_blueprints(text_compiler=agent.rule_based_parse)
        first["egfr"]["parameters"]["kR"] = -999.0
        second = preset_loader.load_blueprints(text_compiler=agent.rule_based_parse)
        self.assertNotEqual(-999.0, second["egfr"]["parameters"]["kR"],
                            "load_blueprints handed out a shared mutable dict, so one "
                            "test's slider sweep could corrupt another's model")
        self.assertIsNot(BLUEPRINTS["egfr"], PRESETS["egfr"]["blueprint"])

    # -- what it refuses -----------------------------------------------------
    def test_malformed_input_raises_instead_of_returning_empty(self):
        for label, bad in self.MALFORMED_LITERALS:
            with self.subTest(malformed=label):
                with self.assertRaises(preset_loader.PresetLoadError) as caught:
                    result = preset_loader.parse_object_literal(bad, origin="sample.js")
                    self.fail("%s parsed to %r instead of raising -- a loader that "
                              "silently returns something for malformed input makes "
                              "every preset test vacuous" % (label, result))
                self.assertTrue(str(caught.exception).strip(),
                                "%s raised with an empty message" % label)

    def test_a_parse_error_names_the_line_and_column(self):
        with self.assertRaises(preset_loader.PresetLoadError) as caught:
            preset_loader.parse_object_literal("{\n  ok: 1,\n  bad: WHAT\n}",
                                               origin="sample.js")
        self.assertRegex(str(caught.exception), r"sample\.js:3:\d+",
                         "a parse failure must say where in the file it happened")

    def test_a_missing_preset_table_raises(self):
        broken = self.source.replace("const Presets = {", "const NotPresets = {", 1)
        with self.assertRaises(preset_loader.PresetLoadError) as caught:
            preset_loader.load_presets(source=broken)
        self.assertIn("Presets", str(caught.exception))

    def test_a_duplicated_preset_table_raises(self):
        with self.assertRaises(preset_loader.PresetLoadError):
            preset_loader.load_presets(source=self.source + "\nconst Presets = {};\n")

    def test_an_empty_preset_table_raises(self):
        with self.assertRaises(preset_loader.PresetLoadError) as caught:
            preset_loader.load_presets(
                source="const Presets = {};\nconst PaperModels = {};\n")
        self.assertIn("EMPTY", str(caught.exception).upper())

    def test_a_missing_shipped_preset_raises(self):
        renamed = self.source.replace("    oscillator: {", "    oscillatorGONE: {", 1)
        self.assertNotEqual(self.source, renamed, "app.js no longer declares oscillator "
                                                 "the way this test renames it")
        with self.assertRaises(preset_loader.PresetLoadError) as caught:
            preset_loader.load_presets(source=renamed)
        self.assertIn("oscillator", str(caught.exception))

    def test_the_synthetic_stand_in_source_is_itself_loadable(self):
        """Guards the negative tests below: they must fail for the defect they inject,
        not because the stand-in source was never valid."""
        loaded = preset_loader.load_presets(source=self._synthetic_source())
        self.assertEqual(sorted(preset_loader.REQUIRED_PRESETS), sorted(loaded))

    def test_a_blueprint_with_no_nodes_or_no_edges_raises(self):
        cases = {
            "no nodes": '{ blueprint: { nodes: [], edges: [{ source: "A", target: "A" }] } }',
            "no edges": '{ blueprint: { nodes: [{ id: "A" }], edges: [] } }',
            "node with no id": '{ blueprint: { nodes: [{ initial_value: 1.0 }], '
                               'edges: [{ source: "A", target: "A" }] } }',
            "edge with no target": '{ blueprint: { nodes: [{ id: "A" }], '
                                   'edges: [{ source: "A" }] } }',
        }
        for label, body in sorted(cases.items()):
            with self.subTest(defect=label):
                with self.assertRaises(preset_loader.PresetLoadError) as caught:
                    preset_loader.load_presets(source=self._synthetic_source(egfr=body))
                self.assertIn("egfr", str(caught.exception))

    def test_a_preset_with_neither_blueprint_nor_text_raises(self):
        with self.assertRaises(preset_loader.PresetLoadError) as caught:
            preset_loader.load_presets(source=self._synthetic_source(egfr="{ targets: [] }"))
        self.assertIn("egfr", str(caught.exception))

    def test_a_text_only_preset_without_a_compiler_raises(self):
        source = self._synthetic_source(
            egfr='{ text: `A activates B.\nA starts at 1.0.`, targets: [] }')
        # It loads as a preset record...
        self.assertIn("text", preset_loader.load_presets(source=source)["egfr"])
        # ...but it cannot become a blueprint without the parser, and must say so
        # rather than going quietly missing from the result.
        with self.assertRaises(preset_loader.PresetLoadError) as caught:
            preset_loader.load_blueprints(source=source)
        self.assertIn("text_compiler", str(caught.exception))

    def test_a_missing_app_js_raises(self):
        with self.assertRaises(preset_loader.PresetLoadError) as caught:
            preset_loader.load_presets(
                path=os.path.join(REPO_DIR, "static", "no-such-app.js"))
        self.assertIn("no-such-app.js", str(caught.exception))


# ==========================================================================
# COVERAGE: A NEW PRESET CANNOT SLIP THROUGH UNTESTED
# ==========================================================================

BEHAVIOUR_CLASS_SUFFIX = "PresetBehaviourTest"


def behaviour_test_classes():
    """{preset name: test class} for every behavioural class in this module.

    Discovered by naming convention -- ``FoldChangePresetBehaviourTest`` covers
    ``foldchange`` -- so adding a preset to app.js without adding its class is what
    PresetCoverageTest sees.
    """
    found = {}
    for name, obj in sorted(globals().items()):
        if (isinstance(obj, type) and issubclass(obj, unittest.TestCase)
                and name.endswith(BEHAVIOUR_CLASS_SUFFIX)
                and name != BEHAVIOUR_CLASS_SUFFIX):
            found[name[:-len(BEHAVIOUR_CLASS_SUFFIX)].lower()] = obj
    return found


class PresetCoverageTest(unittest.TestCase):
    """Every preset app.js ships must have a behavioural test class here.

    Without this, adding a ninth preset to app.js would silently add an untested
    model: the loader would happily return it, no test would name it, and the suite
    would stay green.
    """

    def test_every_shipped_preset_has_a_behavioural_test_class(self):
        covered = behaviour_test_classes()
        uncovered = sorted(set(PRESETS) - set(covered))
        self.assertEqual(
            [], uncovered,
            "static/app.js ships %s with no behavioural test: add a class named "
            "<Name>%s (e.g. %s%s) asserting what the preset's name promises. Loading "
            "a preset is not testing it." % (
                ", ".join(uncovered), BEHAVIOUR_CLASS_SUFFIX,
                (uncovered[0].capitalize() if uncovered else "Xxx"),
                BEHAVIOUR_CLASS_SUFFIX))

    def test_no_behavioural_class_names_a_preset_app_js_no_longer_ships(self):
        covered = behaviour_test_classes()
        orphans = sorted(set(covered) - set(PRESETS))
        self.assertEqual(
            [], orphans,
            "these behavioural classes name presets static/app.js does not define: %s. "
            "Either the preset was renamed (rename the class with it) or it was "
            "removed (remove the class)." % ", ".join(orphans))

    def test_every_behavioural_class_actually_contains_tests(self):
        empty = sorted(name for name, cls in behaviour_test_classes().items()
                       if not [m for m in dir(cls) if m.startswith("test")])
        self.assertEqual([], empty,
                         "these behavioural classes contain no test methods, so the "
                         "presets they name are covered in name only: %s"
                         % ", ".join(empty))


if __name__ == "__main__":
    unittest.main(verbosity=2)
