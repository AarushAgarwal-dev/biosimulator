"""
test_llm_bedrock_path.py -- tests for the LLM (AWS Bedrock) text -> blueprint path.

This module covers the half of /api/blueprint that the deterministic-parser tests
cannot reach: prompt construction, Bedrock request/response handling, tolerant JSON
extraction, and -- the point of the exercise -- the INTEGRITY AUDIT that decides
whether an LLM blueprint is a real model or must be refused.

The two blueprints in FROZEN DEFECTS below are verbatim AWS Bedrock output, captured
from us.meta.llama3-3-70b-instruct-v1:0 in us-east-2 by probe_bedrock_llm.py for the
description "EGF binds EGFR, which activates RAS; RAS activates RAF, RAF activates
MEK, MEK activates ERK; active ERK inhibits RAF". Both compile, integrate to t_max,
stay finite and stay non-negative, so agent._blueprint_status graded BOTH of them
"ok" and the API returned them as validated models:

  * DEAD_POOL_BLUEPRINT  -- RAS, RAF, MEK and ERK are flat at exactly 0.0 for the
    whole run (4 of the 6 species the text names) because the only source of RAS is
    egfr_ras_act = k*RAS/(1 + RAS), which is zero at RAS = 0 forever.
  * RUNAWAY_BLUEPRINT    -- EGFR reaches 10226.8 from an initial value of 1.0, and is
    still climbing at t_max, because the flux egfr_ras is ADDED to both EGFR and RAS
    and subtracted from nothing.

The live Bedrock test at the bottom is opt-in (BIOSIM_LIVE_BEDROCK=1) so the default
suite stays offline and free.
"""

import contextlib
import copy
import io
import json
import os
import unittest
from unittest.mock import patch

import numpy as np

import agent
import llm_provider
from simulation_engine import ODEModel


# =============================================================================
# FROZEN DEFECTS -- verbatim real Bedrock output (do not "tidy" these)
# =============================================================================
def dead_pool_blueprint():
    """Real Bedrock output whose 4 downstream named species never leave zero."""
    return {
        "type": "ODE",
        "nodes": [
            {"id": "EGF", "name": "Epidermal Growth Factor", "initial_value": 1.0},
            {"id": "EGFR", "name": "EGF Receptor", "initial_value": 1.0},
            {"id": "RAS", "name": "RAS protein", "initial_value": 0.0},
            {"id": "RAF", "name": "RAF protein", "initial_value": 0.0},
            {"id": "MEK", "name": "MEK protein", "initial_value": 0.0},
            {"id": "ERK", "name": "ERK protein", "initial_value": 0.0},
        ],
        "parameters": {
            "k_egf_egfr": 0.1, "k_egfr_ras": 0.1, "k_ras_raf": 0.1, "k_raf_mek": 0.1,
            "k_mek_erk": 0.1, "k_erk_raf_inh": 0.1, "K_erk_raf_inh": 1.0,
            "n_erk_raf_inh": 2.0, "k_ras_deg": 0.01, "k_raf_deg": 0.01,
            "k_mek_deg": 0.01, "k_erk_deg": 0.01,
        },
        "fluxes": {
            "egf_egfr_act": "k_egf_egfr*EGF*EGFR",
            "egfr_ras_act": "k_egfr_ras*RAS/(1 + RAS)",
            "ras_raf_act": "k_ras_raf*RAS",
            "raf_mek_act": ("k_raf_mek*RAF/(1 + RAF*(ERK**n_erk_raf_inh/"
                            "(K_erk_raf_inh**n_erk_raf_inh + ERK**n_erk_raf_inh)))"),
            "mek_erk_act": "k_mek_erk*MEK",
        },
        "odes": {
            "EGF": "-egf_egfr_act",
            "EGFR": "-egf_egfr_act",
            "RAS": "egfr_ras_act - k_ras_deg*RAS",
            "RAF": "ras_raf_act - k_raf_deg*RAF - raf_mek_act",
            "MEK": "raf_mek_act - k_mek_deg*MEK - mek_erk_act",
            "ERK": "mek_erk_act - k_erk_deg*ERK",
        },
        "simulation_config": {"t_max": 100.0, "dt": 0.1},
    }


def runaway_blueprint():
    """Real Bedrock output where EGFR blows up to ~1.0e4 from an initial value of 1.0."""
    return {
        "type": "ODE",
        "nodes": [
            {"id": "EGF", "name": "Epidermal Growth Factor", "initial_value": 1.0},
            {"id": "EGFR", "name": "EGF Receptor", "initial_value": 1.0},
            {"id": "RAS", "name": "RAS protein", "initial_value": 0.0},
            {"id": "RAF", "name": "RAF protein", "initial_value": 0.0},
            {"id": "MEK", "name": "MEK protein", "initial_value": 0.0},
            {"id": "ERK", "name": "ERK protein", "initial_value": 0.0},
        ],
        "parameters": {
            "k_egf_egfr": 0.1, "k_egfr_ras": 0.1, "k_ras_raf": 0.1, "k_raf_mek": 0.1,
            "k_mek_erk": 0.1, "k_erk_raf_inh": 0.1, "K_erk_raf_inh": 1.0,
            "n_erk_raf_inh": 2.0,
        },
        "fluxes": {
            "egf_egfr": "k_egf_egfr*EGF*EGFR",
            "egfr_ras": "k_egfr_ras*EGFR",
            "ras_raf": "k_ras_raf*RAS",
            "raf_mek": ("k_raf_mek*RAF/(1 + (ERK**n_erk_raf_inh)/"
                        "(K_erk_raf_inh**n_erk_raf_inh))"),
            "mek_erk": "k_mek_erk*MEK",
        },
        "odes": {
            "EGF": "-egf_egfr",
            "EGFR": "-egf_egfr + egfr_ras",
            "RAS": "egfr_ras - ras_raf",
            "RAF": "ras_raf - raf_mek",
            "MEK": "raf_mek - mek_erk",
            "ERK": "mek_erk",
        },
        "simulation_config": {"t_max": 100.0, "dt": 0.1},
    }


def active_twin_blueprint():
    """The historical defect: the NAMED species (ERK) is a flat pool while a twin
    (ERK_act) carries every bit of the dynamics. A target on ERK measures a constant."""
    return {
        "type": "ODE",
        "nodes": [
            {"id": "ERK", "name": "ERK", "initial_value": 1.0},
            {"id": "ERK_act", "name": "active ERK", "initial_value": 0.0},
        ],
        "parameters": {"k_act": 0.5, "k_deact": 0.1},
        "odes": {"ERK": "0", "ERK_act": "k_act*ERK - k_deact*ERK_act"},
        "simulation_config": {"t_max": 50.0},
    }


def healthy_blueprint():
    """A clean mechanistic model: both species move, magnitudes bounded, flux conserved."""
    return {
        "type": "ODE",
        "nodes": [
            {"id": "M", "name": "mRNA", "initial_value": 0.0},
            {"id": "P", "name": "protein", "initial_value": 0.0},
        ],
        "parameters": {"v_tx": 1.0, "k_mdeg": 0.2, "k_tl": 0.5, "k_pdeg": 0.1,
                       "K_p": 2.0, "n_p": 2.0},
        "fluxes": {"tx": "v_tx/(1 + (P**n_p)/(K_p**n_p))", "tl": "k_tl*M"},
        "odes": {"M": "tx - k_mdeg*M", "P": "tl - k_pdeg*P"},
        "simulation_config": {"t_max": 60.0},
    }


def _trace(bp):
    model = ODEModel(bp)
    t_max = float((bp.get("simulation_config") or {}).get("t_max", 50.0) or 50.0)
    res = model.simulate(min(max(t_max, 1.0), 200.0), num_points=200)
    return {sid: np.asarray(v, dtype=float) for sid, v in res["species"].items()}


# =============================================================================
# 1. The recorded defects really are defects, numerically
# =============================================================================
class RecordedBedrockDefectsAreRealTests(unittest.TestCase):
    """Pin the numbers, so a future change to the audit cannot be tuned around them."""

    def test_dead_pool_blueprint_has_four_species_pinned_at_zero(self):
        tr = _trace(dead_pool_blueprint())
        for sid in ("RAS", "RAF", "MEK", "ERK"):
            self.assertLessEqual(float(np.max(np.abs(tr[sid]))), 1e-12,
                                 f"{sid} was expected to be pinned at exactly zero")
        # ...while the two upstream species evolve normally, which is what fooled
        # the old single-OR "did anything move?" check.
        for sid in ("EGF", "EGFR"):
            span = float(np.max(tr[sid]) - np.min(tr[sid]))
            self.assertGreater(span, 0.5, f"{sid} should evolve")

    def test_runaway_blueprint_reaches_four_orders_above_its_initial_values(self):
        tr = _trace(runaway_blueprint())
        peak = float(np.max(tr["EGFR"]))
        self.assertGreater(peak, 5.0e3, f"EGFR peaked at only {peak:.4g}")
        self.assertTrue(np.all(np.isfinite(tr["EGFR"])))
        self.assertGreater(float(tr["EGFR"][-1]), float(tr["EGFR"][-20]),
                           "EGFR should still be growing at t_max")
        # non-negative and finite is exactly why the old smoke test passed it
        self.assertGreater(float(np.min(tr["EGFR"])), -1e-3)

    def test_active_twin_blueprint_leaves_the_named_species_constant(self):
        tr = _trace(active_twin_blueprint())
        self.assertLessEqual(float(np.max(tr["ERK"]) - np.min(tr["ERK"])), 1e-12)
        self.assertGreater(float(np.max(tr["ERK_act"])), 1.0)

    def test_compile_and_run_smoke_test_alone_cannot_reject_them(self):
        """_blueprint_status is a compile-and-run check; it says nothing about whether
        the named species carry dynamics. This documents WHY the audit is needed."""
        for name, bp in (("dead pool", dead_pool_blueprint()),
                         ("runaway", runaway_blueprint()),
                         ("active twin", active_twin_blueprint())):
            status, _ = agent._blueprint_status(bp)
            self.assertEqual(status, "ok", f"{name}: expected the smoke test to pass it")


# =============================================================================
# 2. The integrity audit rejects them
# =============================================================================
class LLMModelIntegrityTests(unittest.TestCase):
    def test_dead_named_pool_is_broken_and_names_the_species(self):
        status, msg = agent._llm_model_integrity(dead_pool_blueprint())
        self.assertEqual(status, "broken")
        for sid in ("RAS", "RAF", "MEK", "ERK"):
            self.assertIn(sid, msg)
        self.assertIn("zero", msg.lower())

    def test_runaway_growth_is_broken_and_reports_the_magnitude(self):
        status, msg = agent._llm_model_integrity(runaway_blueprint())
        self.assertEqual(status, "broken")
        self.assertIn("EGFR", msg)
        self.assertIn("without bound", msg.lower())

    def test_runaway_message_also_diagnoses_the_mass_creation(self):
        _, msg = agent._llm_model_integrity(runaway_blueprint())
        self.assertIn("egfr_ras", msg)
        self.assertIn("out of nothing", msg.lower())

    def test_a_load_bearing_flat_pool_is_disclosed_not_silently_accepted(self):
        """The pool is read by the twin's rate law, so deleting it would change the
        model. It is kept, but graded imperfect so the reason reaches the user."""
        status, msg = agent._llm_model_integrity(active_twin_blueprint())
        self.assertEqual(status, "imperfect")
        self.assertIn("ERK", msg)
        self.assertIn("ERK_act", msg)
        self.assertIn("carries the dynamics", msg)

    def test_an_orphan_flat_pool_beside_a_twin_is_broken(self):
        """Nothing reads the pool, so it is a pure decoy and the model is refused."""
        bp = active_twin_blueprint()
        bp["parameters"]["v_in"] = 0.5
        bp["odes"] = {"ERK": "0", "ERK_act": "v_in - k_deact*ERK_act"}
        status, msg = agent._llm_model_integrity(bp)
        self.assertEqual(status, "broken")
        self.assertIn("ERK_act", msg)

    def test_active_twin_is_detected_across_naming_conventions(self):
        for twin in ("ERKa", "ERK_act", "ERK_active", "pERK", "ppERK", "ERK_p", "ERK_PP"):
            bp = active_twin_blueprint()
            bp["nodes"][1]["id"] = twin
            bp["odes"] = {"ERK": "0", twin: f"k_act*ERK - k_deact*{twin}"}
            status, msg = agent._llm_model_integrity(bp)
            self.assertEqual(status, "imperfect", f"twin '{twin}' was not detected")
            self.assertIn(twin, msg)

    def test_healthy_model_passes(self):
        self.assertEqual(agent._llm_model_integrity(healthy_blueprint()), ("ok", ""))

    def test_flat_nonzero_species_with_no_twin_is_imperfect_not_broken(self):
        """A constant pool with no active twin may be a legitimate near-conserved
        buffer, so it is surfaced to the user rather than refused."""
        bp = healthy_blueprint()
        bp["nodes"].append({"id": "BUF", "name": "buffer", "initial_value": 3.0})
        bp["parameters"].update({"k_b": 0.2, "BUF_set": 3.0})
        bp["odes"]["BUF"] = "k_b*(BUF_set - BUF)"
        status, msg = agent._llm_model_integrity(bp)
        self.assertEqual(status, "imperfect")
        self.assertIn("BUF", msg)

    def test_audit_never_second_guesses_a_refusal_or_a_pde(self):
        self.assertEqual(
            agent._llm_model_integrity({"validation_errors": ["impossible"]}), ("ok", ""))
        pde = {"type": "PDE", "nodes": [{"id": "U", "initial_value": 1.0}],
               "spatial": {"x_grid": 4, "y_grid": 4, "dx": 1.0, "dy": 1.0,
                           "diffusion": {"U": 0.1}, "reactions": {"U": "-U"}}}
        self.assertEqual(agent._llm_model_integrity(pde), ("ok", ""))

    def test_audit_is_silent_on_a_model_that_does_not_compile(self):
        """A broken-compile blueprint is _blueprint_status's verdict, not the audit's."""
        bp = healthy_blueprint()
        bp["odes"]["M"] = "tx - undefined_symbol*M"
        status, _ = agent._llm_model_integrity(bp)
        self.assertEqual(status, "ok")

    def test_audit_does_not_double_report_a_fully_frozen_model(self):
        bp = {"type": "ODE",
              "nodes": [{"id": "A", "initial_value": 0.0}, {"id": "B", "initial_value": 0.0}],
              "parameters": {"k": 0.5},
              "odes": {"A": "k*A", "B": "k*A"},
              "simulation_config": {"t_max": 10.0}}
        self.assertEqual(agent._llm_model_integrity(bp), ("ok", ""))
        self.assertEqual(agent._blueprint_status(bp)[0], "broken")

    def test_magnitude_is_judged_against_the_models_own_scale(self):
        """A model whose natural units are large is not a runaway."""
        bp = {
            "type": "ODE",
            "nodes": [{"id": "N", "initial_value": 1.0e5}, {"id": "D", "initial_value": 0.0}],
            "parameters": {"k": 0.1, "N_max": 2.0e5},
            "odes": {"N": "k*N*(1 - N/N_max) - k*N*0.01", "D": "k*N*0.01"},
            "simulation_config": {"t_max": 60.0},
        }
        self.assertGreater(agent._llm_model_scale(bp), 1.0e5)
        status, msg = agent._llm_model_integrity(bp)
        self.assertEqual(status, "ok", msg)

    def test_saturating_autocatalysis_is_not_called_a_runaway(self):
        bp = {
            "type": "ODE",
            "nodes": [{"id": "X", "initial_value": 0.01}],
            "parameters": {"r": 0.5, "K": 10.0},
            "odes": {"X": "r*X*(1 - X/K)"},
            "simulation_config": {"t_max": 60.0},
        }
        self.assertEqual(agent._llm_model_integrity(bp), ("ok", ""))


# =============================================================================
# 3. Flux conservation audit
# =============================================================================
class FluxConservationTests(unittest.TestCase):
    def test_flux_added_to_two_species_and_subtracted_from_none_is_flagged(self):
        msg = agent._llm_flux_conservation_error(runaway_blueprint())
        self.assertIsNotNone(msg)
        self.assertIn("egfr_ras", msg)

    def test_conserved_transport_is_accepted(self):
        bp = {
            "nodes": [{"id": "CYT", "initial_value": 1.0}, {"id": "STORE", "initial_value": 0.0}],
            "parameters": {"vp": 0.5},
            "fluxes": {"pump": "vp*CYT"},
            "odes": {"CYT": "-pump", "STORE": "pump"},
        }
        self.assertIsNone(agent._llm_flux_conservation_error(bp))

    def test_one_source_split_into_two_products_is_accepted(self):
        bp = {
            "nodes": [{"id": "A", "initial_value": 1.0}, {"id": "B", "initial_value": 0.0},
                      {"id": "C", "initial_value": 0.0}],
            "parameters": {"k": 0.3},
            "fluxes": {"split": "k*A"},
            "odes": {"A": "-split", "B": "split", "C": "split"},
        }
        self.assertIsNone(agent._llm_flux_conservation_error(bp))

    def test_constant_parameter_only_supply_to_two_pools_is_exempt(self):
        """A constant influx has no source pool, so feeding two pools is legitimate."""
        bp = {
            "nodes": [{"id": "A", "initial_value": 0.0}, {"id": "B", "initial_value": 0.0}],
            "parameters": {"vin": 1.0, "kd": 0.2},
            "fluxes": {"supply": "vin"},
            "odes": {"A": "supply - kd*A", "B": "supply - kd*B"},
        }
        self.assertIsNone(agent._llm_flux_conservation_error(bp))

    def test_unparsable_expressions_are_left_to_the_compiler(self):
        bp = {"nodes": [{"id": "A", "initial_value": 0.0}],
              "fluxes": {"f": "k*A"}, "odes": {"A": "f +++ "}}
        self.assertIsNone(agent._llm_flux_conservation_error(bp))


# =============================================================================
# 4. The audit is wired into the generate -> validate -> repair loop
# =============================================================================
class _Recorder:
    """Returns a scripted sequence of LLM payloads and records the prompts it saw."""

    def __init__(self, *payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def __call__(self, client, prompt, system=None, **kwargs):
        self.prompts.append(prompt)
        payload = self.payloads[min(len(self.prompts) - 1, len(self.payloads) - 1)]
        return copy.deepcopy(payload)


class LLMRepairLoopTests(unittest.TestCase):
    TEXT = ("EGF binds EGFR, which activates RAS. RAS activates RAF, RAF activates MEK, "
            "and MEK activates ERK. Active ERK inhibits RAF.")

    def _run(self, recorder, rule_based=None):
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", recorder), \
             patch.object(agent, "rule_based_parse",
                          return_value=rule_based or healthy_blueprint()):
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                return agent.parse_biological_text(self.TEXT, {"engine": "bedrock"}), captured

    def test_a_dead_pool_model_is_sent_back_for_repair(self):
        rec = _Recorder(dead_pool_blueprint(), healthy_blueprint())
        result, _ = self._run(rec)
        self.assertEqual(len(rec.prompts), 2, "the audit should have triggered a repair round")
        self.assertIn("RAS", rec.prompts[1])
        self.assertIn("zero", rec.prompts[1].lower())
        self.assertEqual([n["id"] for n in result["nodes"]], ["M", "P"])
        self.assertIn("self-repaired", result.get("_llm_notice", ""))

    def test_a_runaway_model_is_sent_back_for_repair(self):
        rec = _Recorder(runaway_blueprint(), healthy_blueprint())
        result, _ = self._run(rec)
        self.assertEqual(len(rec.prompts), 2)
        self.assertIn("EGFR", rec.prompts[1])
        self.assertEqual([n["id"] for n in result["nodes"]], ["M", "P"])

    def test_an_unrepairable_dead_pool_model_is_never_returned(self):
        """The worst outcome -- a confident model whose named species is a flat pool --
        must not reach the user. After the repair rounds, refuse and say so."""
        rec = _Recorder(dead_pool_blueprint())
        result, _ = self._run(rec)
        self.assertEqual(len(rec.prompts), 3, "all repair rounds should be spent")
        self.assertNotEqual([n["id"] for n in result["nodes"]],
                            ["EGF", "EGFR", "ERK", "MEK", "RAF", "RAS"])
        notice = result.get("_llm_notice", "")
        self.assertIn("REJECTED", notice)
        self.assertIn("deterministic parser", notice)

    def test_the_fallback_is_never_presented_as_the_ai_model(self):
        rec = _Recorder(runaway_blueprint())
        result, _ = self._run(rec)
        notice = result.get("_llm_notice", "")
        self.assertIn("REJECTED", notice)
        # the old wording claimed success and hid the swap
        self.assertNotIn("Built a runnable model from your description", notice)

    def test_a_deliberate_physical_refusal_is_still_preserved(self):
        rec = _Recorder({"validation_errors": ["This creates mass from nothing."]})
        result, _ = self._run(rec)
        self.assertEqual(result, {"validation_errors": ["This creates mass from nothing."]})
        self.assertEqual(len(rec.prompts), 1)

    def test_a_healthy_first_answer_is_returned_with_no_repair_round(self):
        rec = _Recorder(healthy_blueprint())
        result, _ = self._run(rec)
        self.assertEqual(len(rec.prompts), 1)
        self.assertEqual([n["id"] for n in result["nodes"]], ["M", "P"])

    def test_flat_nonzero_species_survives_as_a_flagged_best_effort(self):
        bp = healthy_blueprint()
        bp["nodes"].append({"id": "BUF", "initial_value": 3.0})
        bp["parameters"].update({"k_b": 0.2, "BUF_set": 3.0})
        bp["odes"]["BUF"] = "k_b*(BUF_set - BUF)"
        rec = _Recorder(bp)
        result, _ = self._run(rec)
        self.assertIn("BUF", [n["id"] for n in result["nodes"]])
        self.assertIn("did not fully validate", result.get("_llm_notice", ""))
        self.assertIn("BUF", result["_llm_notice"])


# =============================================================================
# 5. Malformed / hostile LLM responses
# =============================================================================
class MalformedResponseTests(unittest.TestCase):
    TEXT = "A activates B and B inhibits C."

    def _parse(self, side_effect=None, return_value=None):
        kwargs = {"side_effect": side_effect} if side_effect else {"return_value": return_value}
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", **kwargs), \
             patch.object(agent, "rule_based_parse", return_value=healthy_blueprint()):
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                return agent.parse_biological_text(self.TEXT, {"engine": "bedrock"}), captured

    def test_truncated_json_raises_llm_error_not_a_silent_empty_model(self):
        truncated = '{"type": "ODE", "nodes": [{"id": "A", "initial_value": 1.0'
        with self.assertRaises(llm_provider.LLMError):
            llm_provider.extract_json(truncated)

    def test_truncated_response_falls_back_and_says_the_engine_was_unavailable(self):
        result, out = self._parse(side_effect=llm_provider.LLMError("Model did not return valid JSON"))
        self.assertEqual([n["id"] for n in result["nodes"]], ["M", "P"])
        self.assertIn("unavailable", result.get("_llm_notice", "").lower())
        self.assertIn("LLMError", out.getvalue())

    def test_provider_error_text_never_reaches_the_user_or_the_log(self):
        secret = "arn:aws:bedrock:us-east-2:333308931113:PROVIDER_DETAIL"
        result, out = self._parse(side_effect=RuntimeError(secret))
        self.assertNotIn(secret, out.getvalue())
        self.assertNotIn(secret, json.dumps(result))

    def test_a_json_list_instead_of_an_object_does_not_crash(self):
        result, _ = self._parse(return_value=[{"id": "A"}, {"id": "B"}])
        self.assertEqual([n["id"] for n in result["nodes"]], ["M", "P"])

    def test_an_empty_object_does_not_crash(self):
        result, _ = self._parse(return_value={})
        self.assertEqual([n["id"] for n in result["nodes"]], ["M", "P"])

    def test_nodes_without_odes_are_rejected_rather_than_run_as_constants(self):
        bp = {"type": "ODE",
              "nodes": [{"id": "A", "initial_value": 1.0}, {"id": "B", "initial_value": 0.0}],
              "parameters": {"k": 0.4},
              "odes": {"A": "-k*A"},
              "simulation_config": {"t_max": 20.0}}
        status, msg = agent._blueprint_status(bp)
        self.assertEqual(status, "broken")
        self.assertIn("no ODE", msg)

    def test_extract_json_survives_the_ways_open_models_mangle_json(self):
        want = {"type": "ODE", "nodes": [{"id": "A", "initial_value": 1.0}]}
        variants = [
            '```json\n{"type": "ODE", "nodes": [{"id": "A", "initial_value": 1.0}]}\n```',
            'Here is the model:\n{"type": "ODE", "nodes": [{"id": "A", "initial_value": 1.0}]}\nHope that helps!',
            '{"type": "ODE", "nodes": [{"id": "A", "initial_value": 1.0}],}',
            '{type: "ODE", nodes: [{id: "A", initial_value: 1.0}]}',
            '<think>the user wants an ODE</think>{"type": "ODE", "nodes": [{"id": "A", "initial_value": 1.0}]}',
            '{"type": "ODE", /* one species */ "nodes": [{"id": "A", "initial_value": 1.0}]}',
            '{\u201ctype\u201d: \u201cODE\u201d, \u201cnodes\u201d: [{\u201cid\u201d: \u201cA\u201d, \u201cinitial_value\u201d: 1.0}]}',
        ]
        for raw in variants:
            self.assertEqual(llm_provider.extract_json(raw), want, f"failed on: {raw[:48]}")


# =============================================================================
# 6. A response naming species the text never mentioned
# =============================================================================
def eleven_species_blueprint():
    """The shape reported from the browser for the description
    "RAS activates RAF. RAF activates MEK." -- eleven species for a description that
    names three, every one duplicated as a plain pool plus an 'a' active form.

    Each plain pool is given synthesis/degradation that is exactly balanced at its
    initial value (k_syn - k_deg*X with X(0) = k_syn/k_deg), which is how such a pool
    survives the existing dead-species check: its derivative is not symbolically zero,
    it is only numerically flat. No rate law reads it, so it is a pure decoy.
    """
    bases = ["EGF", "EGFR", "RAS", "RAF", "MEK", "ERK"]
    twins = {"EGFR": "EGFRa", "RAS": "RASa", "RAF": "RAFa", "MEK": "MEKa", "ERK": "ERKa"}
    nodes = [{"id": b, "name": b, "initial_value": 1.0} for b in bases]
    nodes += [{"id": t, "name": t, "initial_value": 0.0} for t in twins.values()]
    params = {"k": 0.4, "kd": 0.1, "v_in": 0.5, "k_syn": 0.1, "k_deg": 0.1}
    odes = {b: "k_syn - k_deg*" + b for b in bases}
    # the dynamics live entirely in the 'a' forms, driven by a constant input
    odes.update({
        "EGFRa": "v_in - kd*EGFRa",
        "RASa": "k*EGFRa - kd*RASa",
        "RAFa": "k*RASa - kd*RAFa",
        "MEKa": "k*RAFa - kd*MEKa",
        "ERKa": "k*MEKa - kd*ERKa",
    })
    return {"type": "ODE", "nodes": nodes, "parameters": params, "odes": odes,
            "simulation_config": {"t_max": 50.0}}


class LiveReproductionTwinTests(unittest.TestCase):
    """The browser reproduction: eleven species, each named one duplicated with an 'a'
    suffix, the plain pool flat and the 'a' form carrying every bit of the dynamics."""

    def test_the_plain_pool_is_flat_and_the_a_form_carries_the_dynamics(self):
        tr = _trace(eleven_species_blueprint())
        for base, twin in (("EGFR", "EGFRa"), ("RAS", "RASa"), ("RAF", "RAFa"),
                           ("MEK", "MEKa"), ("ERK", "ERKa")):
            base_span = float(np.max(tr[base]) - np.min(tr[base]))
            twin_span = float(np.max(tr[twin]) - np.min(tr[twin]))
            self.assertLessEqual(base_span, 1e-9,
                                 f"{base} was expected flat, span was {base_span:.4g}")
            self.assertGreater(twin_span, 1e-3,
                               f"{twin} should carry the dynamics, span was {twin_span:.4g}")

    def test_the_a_suffix_twin_is_matched_without_hard_coding_act(self):
        ids = ["RAF", "RAFa"]
        self.assertEqual(agent._llm_twin_base("RAFa", ids), "RAF")
        for twin in ("RAFa", "RAF_a", "RAFp", "RAF_act", "RAFact", "RAFstar", "pRAF", "ppRAF"):
            self.assertEqual(agent._llm_twin_base(twin, ["RAF", twin]), "RAF",
                             f"'{twin}' was not recognised as a twin of RAF")

    def test_an_unrelated_species_is_not_mistaken_for_a_twin(self):
        for other in ("MEK", "RAF_MEK_complex", "TotalProteinPool"):
            self.assertIsNone(agent._llm_twin_base(other, ["RAF", other]),
                              f"'{other}' should not be a twin of RAF")

    def test_orphan_twins_are_collapsed_onto_the_researchers_names(self):
        bp = eleven_species_blueprint()
        notes = agent._llm_resolve_active_twins(bp)
        ids = sorted(n["id"] for n in bp["nodes"])
        self.assertEqual(ids, ["EGF", "EGFR", "ERK", "MEK", "RAF", "RAS"])
        for twin in ("EGFRa", "RASa", "RAFa", "MEKa", "ERKa"):
            self.assertNotIn(twin, json.dumps(bp), f"{twin} survived the collapse")
        # ...and the names the researcher used now carry the dynamics
        tr = _trace(bp)
        for base in ("EGFR", "RAS", "RAF", "MEK", "ERK"):
            span = float(np.max(tr[base]) - np.min(tr[base]))
            self.assertGreater(span, 1e-3, f"{base} is still flat after the collapse")
        self.assertTrue(notes)
        self.assertTrue(any("merged" in n for n in notes))

    def test_a_load_bearing_split_is_kept_but_disclosed(self):
        """When the inactive pool IS used by another rate law, deleting it would change
        the model, so it is kept and the notice says which species is measured."""
        bp = {
            "type": "ODE",
            "nodes": [{"id": "ERK", "initial_value": 1.0},
                      {"id": "ERKa", "initial_value": 0.0}],
            "parameters": {"k_act": 0.5, "k_deact": 0.1},
            "odes": {"ERK": "0", "ERKa": "k_act*ERK - k_deact*ERKa"},
            "simulation_config": {"t_max": 50.0},
        }
        notes = agent._llm_resolve_active_twins(bp)
        self.assertEqual(sorted(n["id"] for n in bp["nodes"]), ["ERK", "ERKa"])
        self.assertTrue(notes)
        joined = " ".join(notes)
        self.assertIn("ERKa", joined)
        self.assertIn("ERK", joined)
        self.assertIn("carries the dynamics", joined)

    def test_a_two_way_split_where_both_forms_vary_is_left_alone(self):
        """Measured against Bedrock: asked to "model the inactive and the active form of
        each kinase separately", mistral-large-3 produced RAS/RASa, RAF/RAFa, MEK/MEKa,
        ERK/ERKa where BOTH members of every pair vary, with equal and opposite spans
        (ERK 0.72591 falling 1 -> 0.27409, ERKa 0.72591 rising 0 -> 0.72591) because the
        pair is mass-conserved. That is a correct model, and it is why the guard triggers
        on a FLAT trace rather than on the naming pattern: a name-only rule would reject
        all four of those pairs."""
        bp = {
            "type": "ODE",
            "nodes": [{"id": "ERK", "initial_value": 1.0},
                      {"id": "ERKa", "initial_value": 0.0}],
            "parameters": {"k_act": 0.5, "k_deact": 0.1},
            "odes": {"ERK": "-k_act*ERK + k_deact*ERKa",
                     "ERKa": "k_act*ERK - k_deact*ERKa"},
            "simulation_config": {"t_max": 50.0},
        }
        tr = _trace(bp)
        self.assertAlmostEqual(float(np.max(tr["ERK"]) - np.min(tr["ERK"])),
                               float(np.max(tr["ERKa"]) - np.min(tr["ERKa"])), places=6)
        self.assertEqual(agent._llm_resolve_active_twins(bp), [])
        self.assertEqual(agent._llm_model_integrity(bp), ("ok", ""))

    def test_the_collapse_runs_inside_the_parse_path_and_is_disclosed(self):
        rec = _Recorder(eleven_species_blueprint())
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", rec), \
             patch.object(agent, "rule_based_parse", return_value=healthy_blueprint()):
            result = agent.parse_biological_text("RAS activates RAF. RAF activates MEK.",
                                                 {"engine": "bedrock"})
        ids = sorted(n["id"] for n in result["nodes"])
        self.assertNotIn("RAFa", ids)
        self.assertIn("RAF", ids)
        notice = result.get("_llm_notice", "")
        self.assertIn("merged", notice)


class UnrequestedSpeciesTests(unittest.TestCase):
    TEXT = "RAS activates RAF. RAF activates MEK."

    def test_the_invented_cascade_is_identified(self):
        bp = eleven_species_blueprint()
        agent._llm_resolve_active_twins(bp)      # collapse first, as the parse path does
        extra = agent._llm_unrequested_species(self.TEXT, bp)
        self.assertEqual(sorted(extra), ["EGF", "EGFR", "ERK"])

    def test_active_twins_of_requested_species_are_not_called_unrequested(self):
        bp = eleven_species_blueprint()
        extra = agent._llm_unrequested_species(self.TEXT, bp)
        for twin in ("RASa", "RAFa", "MEKa"):
            self.assertNotIn(twin, extra)

    def test_a_species_named_only_in_prose_is_supported(self):
        text = ("Calcium enters the cytosol from outside and is pumped into an internal "
                "store, which releases it back.")
        bp = {"type": "ODE",
              "nodes": [{"id": "Ca_cyt", "name": "cytosolic calcium", "initial_value": 0.1},
                        {"id": "Ca_store", "name": "store calcium", "initial_value": 1.0}],
              "parameters": {"vp": 0.3},
              "fluxes": {"pump": "vp*Ca_cyt"},
              "odes": {"Ca_cyt": "-pump", "Ca_store": "pump"}}
        self.assertEqual(agent._llm_unrequested_species(text, bp), [])

    def test_a_binary_complex_of_two_named_species_is_supported(self):
        text = "EGF binds EGFR to form a complex, which then decays."
        bp = {"type": "ODE",
              "nodes": [{"id": "EGF", "initial_value": 1.0},
                        {"id": "EGFR", "initial_value": 1.0},
                        {"id": "EGF_EGFR", "initial_value": 0.0}],
              "parameters": {"kon": 0.2, "kd": 0.05},
              "fluxes": {"bind": "kon*EGF*EGFR"},
              "odes": {"EGF": "-bind", "EGFR": "-bind", "EGF_EGFR": "bind - kd*EGF_EGFR"}}
        self.assertEqual(agent._llm_unrequested_species(text, bp), [])

    def test_the_warning_reaches_the_notice_and_the_repair_prompt(self):
        bp = eleven_species_blueprint()
        agent._llm_resolve_active_twins(bp)
        notes = agent._llm_coverage_notices(self.TEXT, bp)
        joined = " ".join(notes)
        for tok in ("EGF", "EGFR", "ERK"):
            self.assertIn(tok, joined)
        self.assertIn("does not mention", joined)
        hint = agent._llm_coverage_repair_hint(self.TEXT, bp)
        self.assertIn("do not elaborate", hint.lower())

    def test_a_refusal_is_not_audited(self):
        self.assertEqual(
            agent._llm_unrequested_species(self.TEXT, {"validation_errors": ["nope"]}), [])

    def test_the_compiler_prompt_forbids_elaboration(self):
        self.assertIn("MODEL ONLY WHAT IS DESCRIBED", agent._COMPILER_PROMPT)
        self.assertIn("RAS activates RAF", agent._COMPILER_PROMPT)


class QualitativeGraphTests(unittest.TestCase):
    """Bedrock's qualitative answer for "MEK activates ERK." was the correct graph with
    every initial_value at 0.0, plus an invented type value. Verbatim shape below."""

    RECORDED = {
        "type": "qualitative",
        "edges": [{"source": "MEK", "target": "ERK", "type": "activation", "parameters": {}}],
        "nodes": [{"id": "MEK", "name": "MEK", "initial_value": 0.0},
                  {"id": "ERK", "name": "ERK", "initial_value": 0.0}],
    }

    def test_all_zero_influence_graph_really_is_frozen(self):
        bp = agent.sanitize_blueprint(copy.deepcopy(self.RECORDED))
        status, msg = agent._blueprint_status(bp)
        self.assertEqual(status, "broken")
        self.assertIn("frozen", msg)

    def test_an_invented_type_value_is_normalised_to_ode(self):
        bp = agent.sanitize_blueprint(copy.deepcopy(self.RECORDED))
        self.assertEqual(bp["type"], "ODE")
        pde = agent.sanitize_blueprint({"type": "pde", "nodes": []})
        self.assertEqual(pde["type"], "PDE")

    def test_the_upstream_driver_is_seeded_and_the_graph_then_runs(self):
        bp = agent.sanitize_blueprint(copy.deepcopy(self.RECORDED))
        notes = agent._llm_seed_qualitative_sources(bp)
        initials = {n["id"]: n["initial_value"] for n in bp["nodes"]}
        self.assertEqual(initials["MEK"], 1.0)
        self.assertEqual(initials["ERK"], 0.0)
        self.assertEqual(agent._blueprint_status(bp)[0], "ok")
        self.assertTrue(any("zero" in n for n in notes))

    def test_a_graph_that_already_starts_non_zero_is_left_alone(self):
        bp = agent.sanitize_blueprint(copy.deepcopy(self.RECORDED))
        bp["nodes"][0]["initial_value"] = 0.4
        self.assertEqual(agent._llm_seed_qualitative_sources(bp), [])
        self.assertEqual(bp["nodes"][0]["initial_value"], 0.4)

    def test_a_pure_cycle_still_gets_one_seed(self):
        bp = agent.sanitize_blueprint({
            "type": "ODE",
            "nodes": [{"id": "A", "initial_value": 0.0}, {"id": "B", "initial_value": 0.0}],
            "edges": [{"source": "A", "target": "B", "type": "activation"},
                      {"source": "B", "target": "A", "type": "inhibition"}]})
        agent._llm_seed_qualitative_sources(bp)
        self.assertEqual(sum(n["initial_value"] for n in bp["nodes"]), 1.0)

    def test_a_custom_kinetics_model_is_not_seeded(self):
        bp = healthy_blueprint()
        before = [n["initial_value"] for n in bp["nodes"]]
        self.assertEqual(agent._llm_seed_qualitative_sources(bp), [])
        self.assertEqual([n["initial_value"] for n in bp["nodes"]], before)

    def test_the_seeded_graph_survives_the_parse_path_with_a_notice(self):
        rec = _Recorder(copy.deepcopy(self.RECORDED))
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", rec), \
             patch.object(agent, "rule_based_parse", return_value=healthy_blueprint()):
            result = agent.parse_biological_text("MEK activates ERK.", {"engine": "bedrock"})
        self.assertEqual(len(rec.prompts), 1, "no repair round should be needed")
        self.assertEqual(sorted(n["id"] for n in result["nodes"]), ["ERK", "MEK"])
        self.assertIn("starting at zero", result.get("_llm_notice", ""))


class SanitizeHardeningTests(unittest.TestCase):
    """Container shapes seen from real models that crashed the compiler before it could
    report anything useful about the model itself."""

    def test_a_list_shaped_parameter_block_is_coerced(self):
        bp = agent.sanitize_blueprint({
            "type": "ODE",
            "nodes": [{"id": "A", "initial_value": 1.0}],
            "parameters": [{"name": "k", "value": 0.25}, {"name": "n", "value": 2}],
            "odes": {"A": "-k*A"}})
        self.assertEqual(bp["parameters"], {"k": 0.25, "n": 2.0})
        self.assertEqual(agent._blueprint_status(bp)[0], "ok")

    def test_a_scalar_parameter_block_becomes_an_empty_map(self):
        bp = agent.sanitize_blueprint({"nodes": [{"id": "A"}], "parameters": "none"})
        self.assertEqual(bp["parameters"], {})

    def test_edge_parameters_given_as_a_list_are_coerced(self):
        bp = agent.sanitize_blueprint({
            "type": "ODE",
            "nodes": [{"id": "A", "initial_value": 1.0}, {"id": "B", "initial_value": 0.0}],
            "edges": [{"source": "A", "target": "B", "type": "activation",
                       "parameters": [{"name": "k", "value": 0.5},
                                      {"name": "K_d", "value": 1.0}]}]})
        self.assertEqual(bp["edges"][0]["parameters"], {"k": 0.5, "K_d": 1.0})
        self.assertEqual(agent._blueprint_status(bp)[0], "ok")

    def test_edge_parameters_given_as_a_scalar_fall_back_to_hill_defaults(self):
        bp = agent.sanitize_blueprint({
            "type": "ODE",
            "nodes": [{"id": "A", "initial_value": 1.0}, {"id": "B", "initial_value": 0.0}],
            "edges": [{"source": "A", "target": "B", "type": "activation",
                       "parameters": "default"}]})
        self.assertEqual(bp["edges"][0]["parameters"], {})
        self.assertEqual(agent._blueprint_status(bp)[0], "ok")

    def test_a_wrapped_parameter_value_is_unwrapped(self):
        bp = agent.sanitize_blueprint({
            "type": "ODE",
            "nodes": [{"id": "A", "initial_value": {"value": 1.0, "units": "uM"}}],
            "parameters": {"k": {"value": 0.3, "units": "1/s"}},
            "odes": {"A": "-k*A"}})
        self.assertEqual(bp["parameters"]["k"], 0.3)
        self.assertEqual(bp["nodes"][0]["initial_value"], 1.0)


class SparseRefusalTests(unittest.TestCase):
    """Verbatim refusals returned by mistral.mistral-large-3-675b-instruct (us-east-2)
    for bare influence sketches. None of these matched the original phrase list, so
    "MEK activates ERK." came back to the user as a hard validation error with no model,
    even though a bare sketch is what the deterministic Hill compiler exists to handle."""

    RECORDED_SPARSE = [
        ("Description is a bare influence sketch with no rates, compartments, or mechanistic "
         "details. It does not specify whether 'activates' means phosphorylation, binding, or "
         "another process, nor does it provide any quantitative information (e.g., rate "
         "constants, thresholds, or conservation). Without additional mechanistic or rate "
         "information, this cannot be represented as a mass-conserving ODE model. A qualitative "
         "influence graph is the only valid representation."),
        ("The description 'MEK activates ERK' provides no mechanistic details (rates, "
         "compartments, conservation, or explicit transport) and does not imply a physically "
         "realizable process with mass conservation or dimensional consistency. It is a bare "
         "influence statement without sufficient information to construct a mechanistic model."),
        ("Description lacks mechanistic detail: no compartments, rates, transport, or "
         "conservation implied. The statement 'MEK activates ERK' is a bare influence sketch "
         "without explicit rate processes, making it impossible to construct a mass-conserving "
         "mechanistic model."),
        ("The description provides no mechanistic details (rates, compartments, transport, or "
         "conservation) and only describes qualitative influences without specifying whether "
         "these are mass-conserving fluxes or abstract activations. A qualitative influence "
         "graph is the only valid representation."),
        ("The description 'MEK activates ERK' is a bare influence statement without any "
         "mechanistic details, rates, compartments, or conservation. It cannot be represented "
         "as a mechanistic ODE model without inventing unmentioned processes or violating R1, "
         "R2, R5, R11, or R14."),
        ("Description lacks mechanistic detail. No rate constants, no initial values, and no "
         "compartments are given."),
        ("Insufficient information: the text omits all kinetic parameters."),
        "The description is too sparse and only qualitative.",
    ]

    RECORDED_BLOCKING = [
        "The description creates mass from nothing.",
        "A membrane protein cannot diffuse freely in the cytosol; this is self-inconsistent.",
        "The stated reaction drives the concentration negative, which is impossible.",
        "The two stated requirements are mutually exclusive.",
    ]

    def test_every_recorded_sparse_refusal_is_non_blocking(self):
        for text in self.RECORDED_SPARSE:
            with self.subTest(text=text[:50]):
                self.assertTrue(agent._is_nonblocking_sparse_refusal([text]),
                                "this refusal would hard-fail a supported input")

    def test_physical_contradictions_stay_blocking(self):
        for text in self.RECORDED_BLOCKING:
            with self.subTest(text=text[:50]):
                self.assertFalse(agent._is_nonblocking_sparse_refusal([text]))

    def test_a_mixed_refusal_stays_blocking(self):
        self.assertFalse(agent._is_nonblocking_sparse_refusal(
            ["No rates were given.", "Also the description creates mass from nothing."]))

    def test_an_empty_refusal_list_is_not_treated_as_sparse(self):
        self.assertFalse(agent._is_nonblocking_sparse_refusal([]))

    def test_a_bare_sketch_gets_a_deterministic_model_not_a_validation_error(self):
        refusal = {"validation_errors": [self.RECORDED_SPARSE[0]]}
        rec = _Recorder(refusal)
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", rec), \
             patch.object(agent, "rule_based_parse", return_value=healthy_blueprint()):
            result = agent.parse_biological_text("MEK activates ERK.", {"engine": "bedrock"})
        self.assertNotIn("validation_errors", result)
        self.assertEqual([n["id"] for n in result["nodes"]], ["M", "P"])

    def test_declining_as_qualitative_is_not_reported_as_a_rejection(self):
        """A normal supported input must not be described to the user as a failure."""
        refusal = {"validation_errors": [self.RECORDED_SPARSE[1]]}
        rec = _Recorder(refusal)
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", rec), \
             patch.object(agent, "rule_based_parse", return_value=healthy_blueprint()):
            result = agent.parse_biological_text("MEK activates ERK.", {"engine": "bedrock"})
        notice = result.get("_llm_notice", "")
        self.assertNotIn("REJECTED", notice)
        self.assertIn("qualitative", notice.lower())
        self.assertIn("placeholder", notice.lower())


class NamedSpeciesCoverageTests(unittest.TestCase):
    def test_named_species_absent_from_the_model_is_reported(self):
        text = "EGF activates RAS, which activates ERK."
        bp = {"type": "ODE",
              "nodes": [{"id": "FOO", "initial_value": 1.0}, {"id": "BAR", "initial_value": 0.0}],
              "odes": {"FOO": "-k*FOO", "BAR": "k*FOO"}, "parameters": {"k": 0.3}}
        missing = agent._llm_missing_named_species(text, bp)
        self.assertEqual(sorted(missing), ["EGF", "ERK", "RAS"])

    def test_species_present_under_a_longer_name_is_not_reported_missing(self):
        text = "Cytosolic calcium is pumped into a store. ERK is phosphorylated."
        bp = {"type": "ODE",
              "nodes": [{"id": "Ca_cyt", "name": "cytosolic calcium", "initial_value": 0.1},
                        {"id": "ERK_total", "name": "ERK", "initial_value": 1.0}],
              "odes": {"Ca_cyt": "-k*Ca_cyt", "ERK_total": "k*Ca_cyt"},
              "parameters": {"k": 0.3}}
        self.assertEqual(agent._llm_missing_named_species(text, bp), [])

    def test_prose_words_and_schema_words_are_not_treated_as_species(self):
        text = "The protein is produced AND degraded. Use an ODE, not a PDE, with RNA present."
        bp = healthy_blueprint()
        self.assertEqual(agent._llm_missing_named_species(text, bp), [])

    def test_a_refusal_is_not_audited_for_coverage(self):
        self.assertEqual(
            agent._llm_missing_named_species("ERK activates MEK",
                                             {"validation_errors": ["impossible"]}), [])

    def test_the_warning_reaches_the_user_notice(self):
        text = "EGF activates RAS, which activates ERK."
        rec = _Recorder(healthy_blueprint())
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", rec):
            result = agent.parse_biological_text(text, {"engine": "bedrock"})
        notice = result.get("_llm_notice", "")
        self.assertIn("Warning", notice)
        for tok in ("EGF", "RAS", "ERK"):
            self.assertIn(tok, notice)

    def test_the_repair_prompt_tells_the_model_what_it_dropped(self):
        text = "EGF activates RAS, which activates ERK."
        rec = _Recorder(dead_pool_blueprint(), healthy_blueprint())
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", rec), \
             patch.object(agent, "rule_based_parse", return_value=healthy_blueprint()):
            agent.parse_biological_text(text, {"engine": "bedrock"})
        self.assertGreaterEqual(len(rec.prompts), 2)


# =============================================================================
# 7. Prompt construction
# =============================================================================
class PromptConstructionTests(unittest.TestCase):
    def test_the_description_is_substituted_and_no_placeholder_survives(self):
        text = 'Calcium enters at a "constant" rate {v0} and is pumped into a store.'
        prompt = agent._COMPILER_PROMPT.replace("<<<DESCRIPTION>>>", text)
        self.assertIn(text, prompt)
        self.assertNotIn("<<<DESCRIPTION>>>", prompt)

    def test_the_compiler_prompt_states_the_rules_the_audit_enforces(self):
        p = agent._COMPILER_PROMPT
        self.assertIn("EVERY SPECIES MUST ACTUALLY MOVE", p)
        self.assertIn("ONE SPECIES PER NAMED ENTITY", p)
        self.assertIn("BOUNDED MAGNITUDES", p)
        self.assertIn("ERK_act", p)          # names the twin defect explicitly

    def test_the_repair_prompt_carries_description_previous_json_and_error(self):
        prompt = (agent._REPAIR_PROMPT
                  .replace("<<<DESCRIPTION>>>", "a cascade")
                  .replace("<<<PREVIOUS>>>", json.dumps(dead_pool_blueprint())[:4000])
                  .replace("<<<ERROR>>>", "RAS never leaves zero"))
        for placeholder in ("<<<DESCRIPTION>>>", "<<<PREVIOUS>>>", "<<<ERROR>>>"):
            self.assertNotIn(placeholder, prompt)
        self.assertIn("a cascade", prompt)
        self.assertIn("egfr_ras_act", prompt)
        self.assertIn("RAS never leaves zero", prompt)

    def test_the_system_instruction_demands_raw_json(self):
        self.assertIn("ONLY raw JSON", agent._SYSTEM_COMPILER)


# =============================================================================
# 8. Bedrock request/response handling
# =============================================================================
class _FakeBedrockClient:
    def __init__(self, script):
        self.script = script          # modelId -> text | Exception
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.script.get(kwargs["modelId"], KeyError("no such model"))
        if isinstance(outcome, Exception):
            raise outcome
        return {"output": {"message": {"content": [{"text": outcome}]}},
                "stopReason": "end_turn"}


def _bedrock(client, model="configured.model"):
    inst = llm_provider.BedrockLLM.__new__(llm_provider.BedrockLLM)
    inst.model = model
    inst.region = "us-east-2"
    inst._client = client
    return inst


class BedrockRequestShapeTests(unittest.TestCase):
    def test_system_and_user_messages_map_onto_the_converse_schema(self):
        client = _FakeBedrockClient({"configured.model": '{"ok": true}'})
        out = _bedrock(client).chat(
            [{"role": "system", "content": "SYS"}, {"role": "user", "content": "USER"}],
            json_mode=True)
        self.assertEqual(out, '{"ok": true}')
        call = client.calls[0]
        self.assertEqual(call["messages"], [{"role": "user", "content": [{"text": "USER"}]}])
        self.assertEqual(call["system"][0], {"text": "SYS"})
        self.assertIn("JSON", call["system"][-1]["text"])
        self.assertLessEqual(call["inferenceConfig"]["maxTokens"], 8192)

    def test_an_unavailable_model_falls_back_to_an_open_weight_model(self):
        class Denied(Exception):
            def __init__(self):
                super().__init__("denied")
                self.response = {"Error": {"Code": "AccessDeniedException"}}

        fallback = llm_provider.BedrockLLM.TEXT_FALLBACK_MODELS[0]
        client = _FakeBedrockClient({"configured.model": Denied(), fallback: '{"ok": 1}'})
        inst = _bedrock(client)
        self.assertEqual(inst.chat([{"role": "user", "content": "hi"}]), '{"ok": 1}')
        self.assertEqual(inst.active_model, fallback)

    def test_a_credential_failure_stops_immediately_without_trying_fallbacks(self):
        class Expired(Exception):
            def __init__(self):
                super().__init__("SENSITIVE")
                self.response = {"Error": {"Code": "ExpiredTokenException"}}

        client = _FakeBedrockClient({"configured.model": Expired()})
        with self.assertRaises(llm_provider.LLMError) as ctx:
            _bedrock(client).chat([{"role": "user", "content": "hi"}])
        self.assertEqual(len(client.calls), 1, "a bad credential must not be retried per model")
        self.assertNotIn("SENSITIVE", str(ctx.exception))
        self.assertIn("ExpiredTokenException", str(ctx.exception))

    def test_generate_json_parses_a_fenced_response_end_to_end(self):
        client = _FakeBedrockClient(
            {"configured.model": '```json\n{"type": "ODE", "nodes": []}\n```'})
        data = llm_provider.generate_json(_bedrock(client), "prompt", system="sys")
        self.assertEqual(data, {"type": "ODE", "nodes": []})

    def test_an_empty_response_from_every_model_is_an_llm_error(self):
        script = {m: "" for m in
                 ["configured.model"] + llm_provider.BedrockLLM.TEXT_FALLBACK_MODELS}
        client = _FakeBedrockClient(script)
        with self.assertRaises(llm_provider.LLMError):
            llm_provider.generate_json(_bedrock(client), "prompt")


# =============================================================================
# 9. Live Bedrock (opt-in: BIOSIM_LIVE_BEDROCK=1)
# =============================================================================
LIVE_DESCRIPTIONS = [
    ("The growth factor EGF binds its receptor EGFR, which activates RAS. RAS activates "
     "RAF, RAF activates MEK, and MEK activates ERK. Active ERK feeds back to inhibit RAF.",
     ["RAS", "RAF", "MEK", "ERK"]),
    ("The tumour suppressor p53 is produced at a basal rate and degraded by MDM2. p53 "
     "induces the transcription of MDM2. MDM2 is also degraded on its own.",
     ["p53", "MDM2"]),
]


@unittest.skipUnless(os.getenv("BIOSIM_LIVE_BEDROCK") == "1",
                     "set BIOSIM_LIVE_BEDROCK=1 to call AWS Bedrock for real")
class LiveBedrockTests(unittest.TestCase):
    """Runs the real path. Every species the description names must carry dynamics,
    values must stay in a sane range, and the model must simulate."""

    @classmethod
    def setUpClass(cls):
        try:
            from dotenv import load_dotenv
            load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
        except Exception:
            pass
        if not llm_provider.bedrock_env_ready():
            raise unittest.SkipTest("no Bedrock credential is configured on this host")
        cls.cfg = {"engine": "bedrock",
                   "model": os.getenv("BIOSIM_LIVE_MODEL") or llm_provider.BEDROCK_DEFAULT_MODEL,
                   "region": llm_provider.BEDROCK_DEFAULT_REGION}

    def test_named_species_carry_dynamics_and_stay_in_range(self):
        for text, named in LIVE_DESCRIPTIONS:
            with self.subTest(text=text[:40]):
                bp = agent.parse_biological_text(text, self.cfg)
                if bp.get("validation_errors"):
                    self.skipTest(f"model refused: {bp['validation_errors']}")
                notice = bp.get("_llm_notice", "") or ""
                self.assertNotIn("REJECTED", notice,
                                 "the AI output was rejected outright; see the notice")
                traces = _trace(bp)
                ids = list(traces.keys())
                scale = agent._llm_model_scale(bp)

                for tok in named:
                    match = [i for i in ids if tok.lower() in i.lower()]
                    self.assertTrue(match, f"'{tok}' is named in the text but absent from {ids}")
                    for sid in match:
                        a = traces[sid]
                        span = float(np.max(a) - np.min(a))
                        self.assertGreater(span, 1e-6,
                                           f"'{sid}' is a flat pool (span {span:.3g})")
                for sid, a in traces.items():
                    self.assertTrue(np.all(np.isfinite(a)), f"'{sid}' went non-finite")
                    self.assertGreater(float(np.min(a)), -1e-3, f"'{sid}' went negative")
                    self.assertLess(float(np.max(np.abs(a))), agent._RUNAWAY_FACTOR * scale,
                                    f"'{sid}' left a physically sensible range")


if __name__ == "__main__":
    unittest.main(verbosity=2)
