"""Focused regressions for backend validation and provider-failure handling."""

import contextlib
import copy
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import agent
import llm_provider
from simulation_engine import ODEModel


def _valid_ode_blueprint():
    return {
        "type": "ODE",
        "nodes": [
            {"id": "A", "initial_value": 1.0},
            {"id": "B", "initial_value": 0.0},
        ],
        "edges": [
            {
                "source": "A",
                "target": "B",
                "type": "activation",
                "parameters": {"k": 0.5, "K_d": 1.0, "n": 1.0},
            }
        ],
        "simulation_config": {"t_max": 1.0},
    }


def _valid_pde_blueprint():
    return {
        "type": "PDE",
        "nodes": [
            {"id": "U", "initial_value": 1.0},
            {"id": "V", "initial_value": 0.5},
        ],
        "spatial": {
            "x_grid": 8,
            "y_grid": 8,
            "dx": 1.0,
            "dy": 1.0,
            "diffusion": {"U": 0.05, "V": 0.2},
            "reactions": {"U": "-U + V", "V": "U - V"},
        },
        "simulation_config": {"t_max": 0.3, "dt": 0.1},
    }


class _BedrockError(Exception):
    def __init__(self, code, message="SENSITIVE_PROVIDER_DETAIL"):
        super().__init__(message)
        self.response = {"Error": {"Code": code, "Message": message}}


def _bedrock_with_client(client):
    instance = llm_provider.BedrockLLM.__new__(llm_provider.BedrockLLM)
    instance.model = "configured.model"
    instance.region = "us-east-2"
    instance._client = client
    return instance


class BlueprintValidationTests(unittest.TestCase):
    def test_physical_refusal_is_preserved(self):
        refusal = {"validation_errors": ["The description creates mass from nothing."]}
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", return_value=refusal):
            result = agent.parse_biological_text("an impossible system", {"engine": "bedrock"})

        self.assertEqual(result, refusal)

    def test_sparse_refusal_uses_deterministic_compiler(self):
        refusal = {"validation_errors": ["The description is too sparse and only qualitative."]}
        fallback = _valid_ode_blueprint()
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", return_value=refusal), \
             patch.object(agent, "rule_based_parse", return_value=fallback):
            result = agent.parse_biological_text("A activates B", {"engine": "bedrock"})

        self.assertNotIn("validation_errors", result)
        self.assertEqual([node["id"] for node in result["nodes"]], ["A", "B"])

    def test_provider_exception_text_is_not_logged_or_returned(self):
        secret = "SENSITIVE_PROVIDER_DETAIL"
        with patch.object(llm_provider, "wants_llm", return_value=True), \
             patch.object(llm_provider, "build_client", return_value=object()), \
             patch.object(llm_provider, "generate_json", side_effect=RuntimeError(secret)), \
             patch.object(agent, "rule_based_parse", return_value=_valid_ode_blueprint()):
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                result = agent.parse_biological_text("A activates B", {"engine": "bedrock"})

        self.assertNotIn(secret, captured.getvalue())
        self.assertNotIn(secret, result.get("_llm_notice", ""))

    def test_pde_smoke_validation_rejects_unknown_symbols(self):
        valid = _valid_pde_blueprint()
        self.assertEqual(agent.validate_blueprint(valid), ("ok", ""))

        invalid = copy.deepcopy(valid)
        invalid["spatial"]["reactions"]["U"] = "U + undeclared_rate"
        status, message = agent.validate_blueprint(invalid)
        self.assertEqual(status, "broken")
        self.assertIn("does not compile or run", message)


class SolverFailureTests(unittest.TestCase):
    def test_unsuccessful_solve_ivp_is_not_returned_as_success(self):
        model = ODEModel(_valid_ode_blueprint())
        failed = SimpleNamespace(
            success=False,
            message="Required step size is less than spacing between numbers.",
            t=np.asarray([0.0]),
            y=np.asarray([[1.0], [0.0]]),
        )
        with patch("simulation_engine.scipy.integrate.solve_ivp", return_value=failed):
            with self.assertRaisesRegex(RuntimeError, "ODE integration failed"):
                model.simulate(t_max=1.0, num_points=5)

    def test_pde_species_named_e_is_bound_as_a_species(self):
        from simulation_engine import solve_pde

        result = solve_pde(
            spatial_config={
                "x_grid": 3, "y_grid": 3, "dx": 1.0, "dy": 1.0,
                "diffusion": {"E": 0.0},
            },
            reaction_formulas={"E": "-E"},
            initial_conditions={"E": {"type": "uniform", "base_value": 1.0}},
            t_max=0.1,
            dt=0.1,
            save_every=1,
        )
        self.assertAlmostEqual(result["species"]["E"][-1][0][0], 0.9)


class BedrockFailureTests(unittest.TestCase):
    def test_auth_failure_stops_before_model_fallback_and_redacts_detail(self):
        class Client:
            def __init__(self):
                self.calls = 0

            def converse(self, **_kwargs):
                self.calls += 1
                raise _BedrockError("ExpiredTokenException")

        client = Client()
        bedrock = _bedrock_with_client(client)
        with self.assertRaises(llm_provider.LLMError) as raised:
            bedrock.chat([{"role": "user", "content": "test"}])

        self.assertEqual(client.calls, 1)
        self.assertIn("authentication failed", str(raised.exception))
        self.assertNotIn("SENSITIVE_PROVIDER_DETAIL", str(raised.exception))

    def test_model_specific_failure_still_uses_fallback(self):
        class Client:
            def __init__(self):
                self.model_ids = []

            def converse(self, **kwargs):
                self.model_ids.append(kwargs["modelId"])
                if len(self.model_ids) == 1:
                    raise _BedrockError("ValidationException")
                return {
                    "output": {"message": {"content": [{"text": "{}"}]}},
                    "stopReason": "end_turn",
                }

        client = Client()
        bedrock = _bedrock_with_client(client)
        self.assertEqual(bedrock.chat([{"role": "user", "content": "test"}]), "{}")
        self.assertEqual(len(client.model_ids), 2)
        self.assertEqual(bedrock.active_model, client.model_ids[1])


class RemoteFailureTests(unittest.TestCase):
    def test_http_error_body_is_not_returned(self):
        secret = "SENSITIVE_PROVIDER_DETAIL"
        response = SimpleNamespace(status_code=500, text=secret)
        remote = llm_provider.RemoteLLM("https://example.invalid/v1", "test-model")

        with patch("requests.post", return_value=response):
            with self.assertRaises(llm_provider.LLMError) as raised:
                remote.chat([{"role": "user", "content": "test"}])

        self.assertIn("HTTP 500", str(raised.exception))
        self.assertNotIn(secret, str(raised.exception))


def _fake_response(status=200, text="", payload=None):
    return SimpleNamespace(status_code=status, text=text, json=lambda: payload)


class BioModelsTests(unittest.TestCase):
    """The live API returns 'format' as a bare string and 'publication' as an
    object; a fixed .get("name") chain raised AttributeError and silently
    substituted offline placeholder data for every live result."""

    def test_string_valued_format_does_not_fall_back_to_offline_data(self):
        import db_interface

        payload = {"models": [{
            "id": "BIOMD0000000628",
            "name": "Li2012 Calcium mediated synaptic plasticity",
            "format": "SBML",                      # a bare string, not {"name": ...}
            "submitter": "Varun Kothamachu",
            "publication": {"title": "Calcium mediated plasticity"},
            "url": "https://www.biomodels.org/BIOMD0000000628",
        }]}
        with patch.object(db_interface.requests, "get", return_value=_fake_response(payload=payload)):
            results = db_interface.search_biomodels("calcium")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "BIOMD0000000628")
        self.assertEqual(results[0]["format"], "SBML")
        self.assertEqual(results[0]["publication"], "Calcium mediated plasticity")

    def test_html_landing_page_is_not_accepted_as_sbml(self):
        import db_interface

        calls = []

        def fake_get(url, **kwargs):
            calls.append(url)
            if len(calls) == 1:
                return _fake_response(text="<!doctype html><html>landing page</html>")
            return _fake_response(text="<?xml version='1.0'?><sbml level='2'><model/></sbml>")

        with patch.object(db_interface.requests, "get", side_effect=fake_get):
            sbml = db_interface.fetch_biomodel_sbml("BIOMD0000000006")

        self.assertIsNotNone(sbml)
        self.assertIn("<sbml", sbml)
        self.assertEqual(len(calls), 2, "should reject the HTML 200 and try the next form")

    def test_zip_payload_is_rejected(self):
        import db_interface

        with patch.object(db_interface.requests, "get", return_value=_fake_response(text="PK\x03\x04binary")):
            self.assertIsNone(db_interface.fetch_biomodel_sbml("BIOMD0000000006"))

    def test_malformed_model_id_never_reaches_the_network(self):
        import db_interface

        with patch.object(db_interface.requests, "get") as get:
            self.assertIsNone(db_interface.fetch_biomodel_sbml("../../etc/passwd"))
        get.assert_not_called()


class MapleValidateTests(unittest.TestCase):
    def test_malformed_target_returns_422_not_500(self):
        from fastapi.testclient import TestClient

        from main import app

        response = TestClient(app).post(
            "/api/maple/validate",
            json={"target_type": "submodel", "target_data": {"nonsense": True}},
        )
        self.assertEqual(response.status_code, 422)
        detail = response.json().get("detail")
        self.assertIsInstance(detail, dict)
        self.assertTrue(detail.get("errors"), "field-level errors should be reported")


if __name__ == "__main__":
    unittest.main()



class TextParserDecimalTests(unittest.TestCase):
    """The deterministic text parser must not truncate decimals.

    Sentences were split on every '.', so "GENB starts at 0.5." became the two
    fragments "GENB starts at 0" and "5" -- silently truncating every decimal to its
    integer part. 0.5 and 0.25 became 0.0, 2.5 became 2.0, and only x.0 values
    appeared to survive, so three of the four shipped text presets ran at initial
    conditions their own descriptions do not specify. That is a wrong simulation with
    no visible symptom, which is exactly the failure class these tests exist for.
    """

    def _initials(self, text):
        import agent
        blueprint = agent.rule_based_parse(text)
        return {node["id"]: node.get("initial_value")
                for node in blueprint.get("nodes", [])}

    def test_sub_unit_initial_values_survive(self):
        got = self._initials("GENA activates GENB. GENA starts at 1.0. "
                             "GENB starts at 0.5.")
        self.assertEqual(got.get("GENA"), 1.0)
        self.assertEqual(got.get("GENB"), 0.5, "0.5 must not be truncated to 0.0")

    def test_a_decimal_above_one_is_not_floored(self):
        # 2.5 -> 2.0 was the tell that this was truncation and not a zeroing bug.
        got = self._initials("A activates B. A starts at 2.5. B starts at 0.25.")
        self.assertEqual(got.get("A"), 2.5)
        self.assertEqual(got.get("B"), 0.25)

    def test_two_decimal_places_survive(self):
        got = self._initials("CAMKII activates CAMKII. CAMKII starts at 0.1.")
        self.assertEqual(got.get("CAMKII"), 0.1)

    def test_sentences_still_split_on_a_real_full_stop(self):
        # The fix must not merge sentences: both relations have to be seen.
        import agent
        blueprint = agent.rule_based_parse(
            "EGF activates AKT. AKT inhibits EGF. EGF starts at 3.0.")
        ids = {node["id"] for node in blueprint.get("nodes", [])}
        self.assertIn("EGF", ids)
        self.assertIn("AKT", ids)
        self.assertGreaterEqual(len(blueprint.get("edges", [])), 2,
                                "both relations must still be parsed")

    def test_a_malformed_number_does_not_raise(self):
        # A stray trailing dot used to reach float() and raise out of the parser.
        try:
            self._initials("A activates B. A starts at 1.0.. B starts at 2.")
        except ValueError as exc:  # pragma: no cover - the point is that it does not
            self.fail(f"the parser raised on a malformed number: {exc}")



class FreeTextSpeciesNameTests(unittest.TestCase):
    """A biologist's own species names must survive the parser.

    This is the path that matters: the presets are demos, but the product's value is a
    researcher describing their own system and getting a model of it. The stopword list
    used to contain PROTEIN, MOLECULE, RECEPTOR, ENZYME, FACTOR and COMPLEX -- every one
    of them a name people actually write. "LIGAND activates RECEPTOR. RECEPTOR activates
    KINASE. KINASE activates TARGET." lost RECEPTOR, which broke the chain, so KINASE and
    TARGET simulated as FLAT LINES while only LIGAND moved. A four-step cascade -- the
    most basic thing anyone types -- silently produced a model missing a step, and a flat
    trace is indistinguishable from a real modelling result.
    """

    def _parse(self, text):
        import agent
        return agent.rule_based_parse(text)

    def _ids(self, text):
        return [node["id"] for node in self._parse(text).get("nodes", [])]

    def test_receptor_is_a_species_not_a_stopword(self):
        ids = self._ids("LIGAND activates RECEPTOR. RECEPTOR activates KINASE. "
                        "LIGAND starts at 5.0.")
        self.assertIn("RECEPTOR", ids,
                      "RECEPTOR is a species name a biologist writes, not a stopword")

    def test_other_biological_nouns_survive(self):
        for name in ("PROTEIN", "ENZYME", "FACTOR", "COMPLEX", "MOLECULE"):
            ids = self._ids(f"UPSTREAM activates {name}. {name} activates DOWNSTREAM. "
                            f"UPSTREAM starts at 1.0.")
            self.assertIn(name, ids, f"{name} must be usable as a species name")

    def test_a_four_step_cascade_keeps_every_step(self):
        text = ("LIGAND activates RECEPTOR. RECEPTOR activates KINASE. "
                "KINASE activates TARGET. LIGAND starts at 5.0. TARGET starts at 0.0.")
        blueprint = self._parse(text)
        ids = {node["id"] for node in blueprint.get("nodes", [])}
        self.assertEqual(ids, {"LIGAND", "RECEPTOR", "KINASE", "TARGET"})
        self.assertGreaterEqual(len(blueprint.get("edges", [])), 3,
                                "all three described relations must be present")

    def test_the_cascade_actually_propagates(self):
        """The real symptom: a dropped species leaves everything downstream flat."""
        import numpy as np
        import simulation_engine
        blueprint = self._parse(
            "LIGAND activates RECEPTOR. RECEPTOR activates KINASE. "
            "KINASE activates TARGET. LIGAND starts at 5.0. TARGET starts at 0.0.")
        result = simulation_engine.ODEModel(blueprint).simulate(t_max=40.0)
        for species in ("RECEPTOR", "KINASE", "TARGET"):
            values = np.asarray(result["species"][species], dtype=float)
            self.assertGreater(
                float(values.max() - values.min()), 1e-6,
                f"{species} never changes, so the cascade is not propagating through it")

    def test_english_function_words_are_still_rejected(self):
        # The fix must not turn every word into a species.
        ids = self._ids("The A activates B and it inhibits A. A starts at 1.0.")
        for word in ("THE", "AND", "IT"):
            self.assertNotIn(word, ids, f"{word!r} is English, not a species")

    def test_a_dropped_relation_endpoint_is_reported(self):
        """If a name IS rejected, the resulting hole must not be silent."""
        blueprint = self._parse("A activates THE. THE activates B. A starts at 1.0.")
        notice = str(blueprint.get("_llm_notice") or "")
        if "THE" not in {n["id"] for n in blueprint.get("nodes", [])}:
            self.assertTrue(notice,
                            "a rejected relation endpoint must produce a notice, because "
                            "the chain is broken and downstream species will be flat")



class ProseDescriptionTests(unittest.TestCase):
    """A researcher writes prose, not the parser's grammar.

    The product's value is a biologist describing their OWN system, so the shape of the
    sentence must not decide whether they get a model. Every case here failed before:
    passive voice ("ERK is phosphorylated by MEK") produced NO species at all;
    "GLUCOSE stimulates release of INSULIN" produced a fabricated species RELEASE and
    silently dropped INSULIN; and a correctly-parsed cascade compiled to a model sitting
    entirely at zero, because nothing seeded it and a Hill term of zero is zero.

    The standard applied throughout: a description either produces a model that MOVES,
    or it says what it could not do. A confident flat model is the one unacceptable
    outcome, because a flat trace is indistinguishable from a real modelling result.
    """

    def _parse(self, text):
        import agent
        return agent.rule_based_parse(text)

    def _ids(self, blueprint):
        return {node["id"] for node in blueprint.get("nodes", [])}

    def _simulate(self, blueprint, t_max=40.0):
        import numpy as np
        import simulation_engine
        result = simulation_engine.ODEModel(blueprint).simulate(t_max=t_max)
        return {sid: np.asarray(vals, dtype=float)
                for sid, vals in result["species"].items()}

    def _assert_moves(self, blueprint, species):
        traces = self._simulate(blueprint)
        for sid in species:
            values = traces[sid]
            self.assertGreater(
                float(values.max() - values.min()), 1e-9,
                f"{sid} is a flat line, which is indistinguishable from a real result")

    def test_passive_voice_builds_the_cascade(self):
        blueprint = self._parse(
            "ERK is phosphorylated by MEK. MEK is activated by RAF. "
            "RAF is turned on by RAS.")
        self.assertEqual(self._ids(blueprint), {"ERK", "MEK", "RAF", "RAS"})
        self._assert_moves(blueprint, ["RAF", "MEK", "ERK"])

    def test_passive_voice_direction_is_not_inverted(self):
        """"ERK is phosphorylated by MEK" is MEK -> ERK, never the reverse."""
        blueprint = self._parse("ERK is phosphorylated by MEK. MEK starts at 1.0.")
        edges = [(e["source"], e["target"]) for e in blueprint["edges"]]
        self.assertIn(("MEK", "ERK"), edges)
        self.assertNotIn(("ERK", "MEK"), edges)

    def test_a_nominalisation_is_not_a_species(self):
        """"release of INSULIN" names a process; the species is INSULIN."""
        blueprint = self._parse(
            "GLUCOSE stimulates release of INSULIN. GLUCOSE starts at 2.0.")
        ids = self._ids(blueprint)
        self.assertIn("INSULIN", ids)
        self.assertNotIn("RELEASE", ids, "a process was fabricated as a species")

    def test_interjections_do_not_take_the_subject_slot(self):
        for text, stray in (
            ("INSULIN in turn lowers GLUCOSE. GLUCOSE starts at 2.0.", "TURN"),
            ("P53 also drives production of MDM2. P53 starts at 1.0.", "ALSO"),
        ):
            ids = self._ids(self._parse(text))
            self.assertNotIn(stray, ids, f"{stray!r} was read as a species in {text!r}")

    def test_a_closed_feedback_loop_is_not_inert(self):
        """Every species has an incoming edge, so nothing was ever seeded."""
        blueprint = self._parse(
            "GLUCOSE stimulates INSULIN. INSULIN lowers GLUCOSE.")
        self.assertEqual(self._ids(blueprint), {"GLUCOSE", "INSULIN"})
        self._assert_moves(blueprint, ["GLUCOSE", "INSULIN"])

    def test_an_only_inhibited_species_can_still_fall(self):
        """Inhibition alone cannot lift a species off zero, so it must start non-zero."""
        blueprint = self._parse("MDM2 degrades P53. MDM2 starts at 1.0.")
        p53 = next(n for n in blueprint["nodes"] if n["id"] == "P53")
        self.assertGreater(float(p53["initial_value"]), 0.0,
                           "a species that is only ever inhibited cannot start at zero")
        self._assert_moves(blueprint, ["P53"])

    def test_a_stated_initial_value_is_never_overwritten(self):
        """Seeding must not touch a number the researcher gave, including zero."""
        blueprint = self._parse(
            "LIGAND activates RECEPTOR. LIGAND starts at 5.0. RECEPTOR starts at 0.0.")
        values = {n["id"]: float(n["initial_value"]) for n in blueprint["nodes"]}
        self.assertEqual(values["LIGAND"], 5.0)
        self.assertEqual(values["RECEPTOR"], 0.0,
                         "an explicit 0.0 was overwritten by the seeding default")

    def test_seeding_is_disclosed(self):
        """A value the product chose must be visible, not hidden in the model."""
        blueprint = self._parse("RAS activates RAF.")
        self.assertTrue(str(blueprint.get("_llm_notice") or ""),
                        "the model silently invented a starting amount")

    def test_arrow_notation(self):
        blueprint = self._parse("A -> B. B -> C. A starts at 1.0.")
        self.assertEqual(self._ids(blueprint), {"A", "B", "C"})
        self._assert_moves(blueprint, ["B", "C"])

    def test_unparseable_text_is_still_refused(self):
        """Widening the grammar must not make it credulous."""
        blueprint = self._parse(
            "The system behaves in an interesting way under stress and then relaxes.")
        self.assertFalse(blueprint.get("nodes"),
                         "a description naming no species produced a model anyway")
        self.assertTrue(str(blueprint.get("_llm_notice") or ""),
                        "the refusal was silent")
