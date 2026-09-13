"""Route hardening: every API defect found by the 56-endpoint sweep, with the
exact request that triggered it.

Each test below carries the payload the sweep sent and the status it got back, so a
regression is visible as a status change rather than as a vague failure. The theme
across all of them is one rule: an endpoint may refuse a request, but it must never
report a run it did not perform. A truncated horizon, a reversed integration, a
defaulted grid, a swapped bound and a fabricated 0.0 all came back as 200, and
nothing downstream -- least of all the closed-loop refinement that grades these
runs -- could tell any of them from a real result.

Runs against the app imported from main via TestClient, so no server is involved.
"""

import time
import unittest

from fastapi.testclient import TestClient

import geometry as geo
import main
from main import app

# raise_server_exceptions=False keeps the global 500 handler in the loop, so a
# regression that reintroduces an unhandled exception shows up as a 500 response
# here rather than as an exception escaping the client.
client = TestClient(app, raise_server_exceptions=False)


# --- shared payloads -------------------------------------------------------

def hill_blueprint():
    """An ODE blueprint compiled through the generic Hill path, so it has real
    generated parameters (syn_*, deg_*, act_*) to fit and to sample."""
    return {
        "nodes": [{"id": "A", "initial_value": 1.0, "synthesis": 0.5, "degradation": 0.1},
                  {"id": "B", "initial_value": 0.5, "degradation": 0.2}],
        "edges": [{"source": "A", "target": "B", "type": "activation",
                   "parameters": {"k": 0.5, "K_d": 1.0, "n": 2.0}}],
        "simulation_config": {"t_max": 10.0},
    }


def pde_blueprint(**config):
    settings = {"t_max": 0.5, "dt": 0.05}
    settings.update(config)
    return {
        "type": "PDE",
        "nodes": [{"id": "U", "initial_value": 1.0}],
        "edges": [],
        "spatial": {"x_grid": 12, "y_grid": 12, "dx": 1.0, "dy": 1.0,
                    "diffusion": {"U": 0.1}, "reactions": {"U": "0.1*U*(1-U)"}},
        "simulation_config": settings,
    }


def abm_blueprint(**config):
    settings = {"num_mcs": 2, "save_every": 1}
    settings.update(config)
    return {
        "grid": {"width": 16, "height": 16},
        "temperature": 10.0,
        "cell_types": [{"type_id": 1, "name": "Cell", "target_volume": 20,
                        "lambda_volume": 2.0, "color": [200, 100, 100]}],
        "initial_config": [{"type_id": 1, "count": 2, "radius": 2, "region": "center"}],
        "simulation_config": settings,
    }


def one_d_field(**overrides):
    field = {"name": "u", "units": "mM", "diffusion": "D", "initial": "0.0",
             "reaction": "0", "source": "0", "advection": {"vx": 0.0, "vy": 0.0},
             "t_start": 0.0, "t_end": 2.0, "output_interval": 0.5}
    field.update(overrides)
    return {"parameters": {"D": 1.0}, "fields": [field]}


def solve_1d(model, **extra):
    payload = {"domain": geo.make_interval(1.0), "model": model,
               "mesh_settings": {"element_count": 20}}
    payload.update(extra)
    return client.post("/api/pde/solve1d", json=payload)


def detail_text(response):
    """The response's detail as a string, whatever shape it came back in."""
    try:
        body = response.json()
    except ValueError:
        return response.text
    detail = body.get("detail", body)
    if isinstance(detail, dict):
        parts = [str(detail.get("message", ""))]
        parts.extend(str(item) for item in detail.get("missing", []))
        parts.extend(str(issue) for issue in detail.get("issues", []))
        return " ".join(parts)
    return str(detail)


# =========================================================================
# P1 -- /api/simulate had no execution budget
# =========================================================================

class ExecutionBudgetTests(unittest.TestCase):
    """One request must not be able to occupy the endpoint indefinitely.

    Measured before: blueprint with odes {"X": "X**X**X"} and t_max=1e9 returned NO
    BYTES within 120 s. num_points was capped at 5000, but the cap is not a budget:
    LSODA's internal step count is driven by the right-hand side, not by the output
    grid, so an explosive system runs unbounded between two output points.
    """

    HANG = {"nodes": [{"id": "X", "initial_value": 1.0}], "edges": [],
            "odes": {"X": "X**X**X"},
            "simulation_config": {"t_max": 1000000000.0}}

    def test_the_measured_hang_payload_is_refused_not_served(self):
        started = time.monotonic()
        response = client.post("/api/simulate", json={"blueprint": self.HANG})
        elapsed = time.monotonic() - started
        self.assertEqual(response.status_code, 400, response.text[:400])
        # t_max=1e9 is over the horizon bound, so this is refused before any
        # integration starts -- it must be fast, not merely finite.
        self.assertLess(elapsed, 15.0)
        self.assertNotIn("species", response.json())

    def test_the_refusal_names_the_limit_and_the_offending_value(self):
        response = client.post("/api/simulate", json={"blueprint": self.HANG})
        detail = detail_text(response)
        self.assertIn("t_max", detail)
        self.assertIn("1e+09", detail)              # the value the caller sent
        self.assertIn(f"{main.MAX_T_MAX:g}", detail)  # the limit it broke

    def test_the_horizon_is_refused_never_silently_shortened(self):
        """A shortened run reported as complete is worse than a refusal."""
        response = client.post("/api/simulate", json={"blueprint": self.HANG})
        self.assertNotEqual(response.status_code, 200)
        self.assertNotIn("species", response.json())

    def test_a_stiff_model_inside_the_horizon_bound_still_hits_the_deadline(self):
        """The horizon bound alone is not enough, so the deadline is tested alone.

        This t_max is well inside MAX_T_MAX, so only the in-integration wall-clock
        check can stop it. The budget is squeezed to 1.5 s for the duration of the
        test so the assertion does not cost 20 s.
        """
        original = main.SIMULATE_BUDGET_SECONDS
        main.SIMULATE_BUDGET_SECONDS = 1.5
        try:
            started = time.monotonic()
            response = client.post("/api/simulate", json={"blueprint": {
                "nodes": [{"id": "X", "initial_value": 1.0}], "edges": [],
                "odes": {"X": "X**X**X"},
                "simulation_config": {"t_max": 100000.0}}})
            elapsed = time.monotonic() - started
        finally:
            main.SIMULATE_BUDGET_SECONDS = original
        self.assertEqual(response.status_code, 400, response.text[:400])
        self.assertLess(elapsed, 20.0)
        detail = detail_text(response)
        self.assertIn("execution budget", detail)
        self.assertIn("ABANDONED", detail)

    def test_the_budget_leaves_the_model_object_untouched(self):
        """The deadline wraps _f_lambdified, so it must restore it on the way out."""
        from simulation_engine import ODEModel
        model = ODEModel(hill_blueprint())
        before = model._f_lambdified
        with main._ode_execution_budget(model, 10.0, budget_seconds=5.0):
            self.assertIsNot(model._f_lambdified, before)
        self.assertIs(model._f_lambdified, before)

    def test_a_normal_model_is_unaffected_by_the_budget(self):
        response = client.post("/api/simulate", json={"blueprint": hill_blueprint()})
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertIn("species", response.json())

    def test_a_pde_horizon_that_needs_too_many_steps_is_refused(self):
        """solve_pde materialises one list entry per step, so t_max/dt is bounded."""
        response = client.post("/api/simulate", json={
            "blueprint": pde_blueprint(t_max=100000.0, dt=1e-06)})
        self.assertEqual(response.status_code, 400, response.text[:300])
        self.assertIn("explicit time steps", detail_text(response))


# =========================================================================
# P2 -- a full default run for an empty request body
# =========================================================================

class EmptyBodyAmplificationTests(unittest.TestCase):
    """Measured before: POST {"blueprint": {}} to /api/abm/simulate returned 200 in
    13.8 s with 5,621,290 bytes -- a 100x100 lattice over 500 Monte Carlo steps.
    /api/multiscale/simulate did the same for 1,212,558 bytes. Both are a success
    reported for input nobody supplied, and an unauthenticated amplification vector:
    a few concurrent 16-byte bodies saturate the process.
    """

    def test_abm_empty_blueprint_is_422_not_a_five_megabyte_run(self):
        response = client.post("/api/abm/simulate", json={"blueprint": {}})
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertLess(len(response.content), 4000)

    def test_abm_empty_blueprint_lists_every_missing_field(self):
        detail = detail_text(client.post("/api/abm/simulate", json={"blueprint": {}}))
        self.assertIn("grid.width", detail)
        self.assertIn("cell_types", detail)
        self.assertIn("initial_config", detail)

    def test_multiscale_empty_blueprint_is_422_not_a_default_run(self):
        response = client.post("/api/multiscale/simulate", json={"abm_blueprint": {}})
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertLess(len(response.content), 4000)

    def test_multiscale_requires_the_step_count_it_used_to_default(self):
        """num_mcs is a REQUEST field here, so an omitted one bought 100 steps."""
        detail = detail_text(client.post("/api/multiscale/simulate",
                                         json={"abm_blueprint": abm_blueprint()}))
        self.assertIn("num_mcs", detail)

    def test_a_lattice_missing_only_its_cell_types_names_only_that(self):
        blueprint = abm_blueprint()
        blueprint.pop("cell_types")
        detail = detail_text(client.post("/api/abm/simulate",
                                         json={"blueprint": blueprint}))
        self.assertIn("cell_types", detail)
        self.assertNotIn("grid.width", detail)

    def test_a_cell_type_without_a_type_id_is_named_by_index(self):
        blueprint = abm_blueprint()
        blueprint["cell_types"] = [{"name": "Nameless"}]
        detail = detail_text(client.post("/api/abm/simulate",
                                         json={"blueprint": blueprint}))
        self.assertIn("cell_types[0].type_id", detail)

    def test_a_fully_specified_abm_request_still_runs(self):
        response = client.post("/api/abm/simulate", json={"blueprint": abm_blueprint()})
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertEqual(len(response.json()["t"]), 3)      # t=0,1,2

    def test_a_fully_specified_multiscale_request_still_runs(self):
        response = client.post("/api/multiscale/simulate", json={
            "abm_blueprint": abm_blueprint(), "num_mcs": 2, "save_every": 1})
        self.assertEqual(response.status_code, 200, response.text[:300])

    def test_every_shipped_abm_preset_satisfies_the_required_fields(self):
        """The requirement must not lock the app out of its own presets."""
        import abm_blueprints
        for name in abm_blueprints.ABM_PRESETS:
            preset = client.get(f"/api/abm/preset/{name}").json()
            self.assertEqual(main._missing_lattice_fields(preset, "blueprint"), [],
                             msg=f"preset {name} would be refused")


# =========================================================================
# P3 -- unhandled exceptions reaching the global 500 handler
# =========================================================================

class MalformedTopologyTests(unittest.TestCase):
    """Measured before: 500 from the global handler for ordinary bad input.

    topology_module.validate_topology already reports exactly these problems --
    /api/topology/validate answers with code `nodes_not_list` -- so the validator
    existed. /api/topology/cycles and /api/topology/export simply never called it,
    and the strings reached adjacency() and nodes_to_csv() as an AttributeError.
    """

    STRING_MEMBERS = {"nodes": "x", "edges": "y", "cells": []}
    STRING_EDGE_LIST = {"nodes": "x", "edges": ["e1"], "cells": []}
    STRING_EVERYTHING = {"nodes": ["n1"], "edges": ["e1"], "cells": ["c1"]}

    def test_cycles_with_string_nodes_and_edges_is_422_not_500(self):
        response = client.post("/api/topology/cycles",
                               json={"topology": self.STRING_MEMBERS})
        self.assertEqual(response.status_code, 422, response.text[:300])

    def test_cycles_with_a_list_of_edge_strings_is_422_not_500(self):
        response = client.post("/api/topology/cycles",
                               json={"topology": self.STRING_EDGE_LIST})
        self.assertEqual(response.status_code, 422, response.text[:300])

    def test_export_with_string_members_is_422_not_500(self):
        response = client.post("/api/topology/export",
                               json={"topology": self.STRING_EVERYTHING})
        self.assertEqual(response.status_code, 422, response.text[:300])

    def test_the_refusal_carries_the_validator_code_not_a_generic_message(self):
        payload = client.post("/api/topology/cycles",
                              json={"topology": self.STRING_MEMBERS}).json()
        codes = [issue["code"] for issue in payload["detail"]["issues"]]
        self.assertIn("nodes_not_list", codes)

    def test_no_topology_route_returns_the_generic_five_hundred_body(self):
        for path in ("/api/topology/validate", "/api/topology/cycles",
                     "/api/topology/export"):
            for topology in (self.STRING_MEMBERS, self.STRING_EDGE_LIST,
                             self.STRING_EVERYTHING):
                response = client.post(path, json={"topology": topology})
                self.assertNotEqual(response.status_code, 500,
                                    msg=f"{path} still 500s on {topology}")

    def test_validate_keeps_reporting_the_problem_with_a_200(self):
        """This route's job is to DIAGNOSE, so it must answer even when statistics
        over non-object members cannot be computed."""
        response = client.post("/api/topology/validate",
                               json={"topology": self.STRING_MEMBERS})
        self.assertEqual(response.status_code, 200, response.text[:300])
        payload = response.json()
        self.assertFalse(payload["valid"])
        self.assertIn("nodes_not_list", [i["code"] for i in payload["issues"]])
        self.assertIn("stats_unavailable", payload)

    def test_a_well_formed_topology_still_exports(self):
        topology = {"nodes": [{"id": "n1", "x": 0.0, "y": 0.0},
                              {"id": "n2", "x": 1.0, "y": 0.0}],
                    "edges": [{"id": "e1", "source": "n1", "target": "n2"}],
                    "cells": []}
        for path in ("/api/topology/cycles", "/api/topology/export"):
            self.assertEqual(client.post(path, json={"topology": topology}).status_code,
                             200, msg=path)


class OneDimensionalStepBudgetTests(unittest.TestCase):
    """Measured before: 500 (MemoryError) for a field with t_end=1e9 and
    output_interval=1e8. solve_1d builds `step_sizes = [dt] * n_full`, so the step
    count is a memory bound: at the stability-limited dt that is ~2e12 entries.
    """

    def test_a_huge_time_window_is_422_not_a_memory_error(self):
        response = solve_1d(one_d_field(t_end=1e9, output_interval=1e8))
        self.assertEqual(response.status_code, 422, response.text[:300])

    def test_the_refusal_names_the_step_count_and_the_limit(self):
        detail = detail_text(solve_1d(one_d_field(t_end=1e9, output_interval=1e8)))
        self.assertIn("explicit time steps", detail)
        self.assertIn(f"{main.MAX_PDE_STEPS:,}", detail)
        self.assertIn("1e+09", detail)

    def test_a_reversed_time_window_is_refused(self):
        """Already handled: pde_model.compile_field refuses this with an actionable
        400, so this is regression cover rather than a new fix."""
        response = solve_1d(one_d_field(t_start=5.0, t_end=1.0))
        self.assertEqual(response.status_code, 400, response.text[:300])
        self.assertIn("must be after the start time", detail_text(response))

    def test_a_normal_one_d_solve_still_works(self):
        response = solve_1d(one_d_field())
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertGreater(len(response.json()["t"]), 1)


# =========================================================================
# P4 -- 500 bodies that are a bare Python repr
# =========================================================================

class ActionableMessageTests(unittest.TestCase):
    """Measured before: {"detail": "'id'"}, {"detail": "'ComplexInfinity'"} and
    {"detail": "'does_not_exist'"} -- a repr of a KeyError argument with no field
    name, no index and nothing a caller can act on, for ordinary bad input.
    """

    NO_ID = {"nodes": [{"initial_value": 1.0}], "edges": []}
    DIVIDE_BY_ZERO = {"nodes": [{"id": "X", "initial_value": 1.0}], "edges": [],
                      "odes": {"X": "1/0"},
                      "simulation_config": {"t_max": 5.0}}

    def test_compile_names_the_node_index_that_has_no_id(self):
        response = client.post("/api/compile", json={"blueprint": self.NO_ID})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("nodes[0]", detail)
        self.assertIn("id", detail)
        self.assertNotEqual(detail.strip(), "'id'")

    def test_simulate_names_the_node_index_that_has_no_id(self):
        response = client.post("/api/simulate", json={"blueprint": self.NO_ID})
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertIn("nodes[0]", detail_text(response))

    def test_the_offending_index_is_the_real_one_not_always_zero(self):
        blueprint = {"nodes": [{"id": "A", "initial_value": 1.0},
                               {"id": "B", "initial_value": 1.0},
                               {"initial_value": 1.0}], "edges": []}
        detail = detail_text(client.post("/api/compile", json={"blueprint": blueprint}))
        self.assertIn("nodes[2]", detail)

    def test_a_divide_by_zero_rate_law_names_the_species(self):
        response = client.post("/api/simulate", json={"blueprint": self.DIVIDE_BY_ZERO})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("'X'", detail)
        self.assertIn("division by zero", detail)
        self.assertNotIn("ComplexInfinity", detail)

    def test_an_unreadable_rate_law_is_422_not_500(self):
        response = client.post("/api/simulate", json={"blueprint": {
            "nodes": [{"id": "X", "initial_value": 1.0}], "edges": [],
            "odes": {"X": "3 +* /"}, "simulation_config": {"t_max": 5.0}}})
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertIn("odes['X']", detail_text(response))

    def test_optimize_names_the_unknown_parameter_and_the_real_ones(self):
        response = client.post("/api/optimize", json={
            "blueprint": hill_blueprint(),
            "target_data": {"A": [1.0, 1.0]}, "target_times": [0.0, 1.0],
            "params_to_fit": ["does_not_exist"]})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("does_not_exist", detail)
        self.assertIn("deg_A", detail)          # a parameter the model really has
        self.assertNotEqual(detail.strip(), "'does_not_exist'")

    def test_optimize_still_fits_a_real_parameter(self):
        response = client.post("/api/optimize", json={
            "blueprint": hill_blueprint(),
            "target_data": {"A": [1.0, 2.0, 3.0]}, "target_times": [0.0, 1.0, 2.0],
            "params_to_fit": ["syn_A"]})
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertIn("syn_A", response.json()["fitted_parameters"])

    def test_a_duplicate_species_id_is_named_rather_than_silently_merged(self):
        response = client.post("/api/compile", json={"blueprint": {
            "nodes": [{"id": "A", "initial_value": 1.0},
                      {"id": "A", "initial_value": 2.0}], "edges": []}})
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertIn("repeats the id", detail_text(response))


# =========================================================================
# P6 -- nonsense numerics accepted and reported as a successful run
# =========================================================================

class NonsenseNumericsTests(unittest.TestCase):
    """Measured before, all 200: t_max=-10 integrated BACKWARDS in time; a PDE grid
    nx=0/ny=0 was silently replaced by a 50x50 default; dt=-0.1 integrated nothing
    and reported success; /api/pde/solve1d with dt=-1.0 returned steps=0 while its
    own summary said "Solved from t=0 to t=10"; num_mcs=-5 returned t=[0].
    """

    def test_a_negative_ode_horizon_is_refused_not_integrated_backwards(self):
        blueprint = hill_blueprint()
        blueprint["simulation_config"]["t_max"] = -10
        response = client.post("/api/simulate", json={"blueprint": blueprint})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("t_max", detail)
        self.assertIn("-10", detail)
        self.assertIn("backwards", detail)

    def test_a_negative_horizon_no_longer_returns_negative_times(self):
        blueprint = hill_blueprint()
        blueprint["simulation_config"]["t_max"] = -10
        response = client.post("/api/simulate", json={"blueprint": blueprint})
        self.assertNotIn("t", response.json())

    def test_a_zero_ode_horizon_is_refused(self):
        blueprint = hill_blueprint()
        blueprint["simulation_config"]["t_max"] = 0
        self.assertEqual(
            client.post("/api/simulate", json={"blueprint": blueprint}).status_code, 422)

    def test_a_non_numeric_horizon_is_refused(self):
        blueprint = hill_blueprint()
        blueprint["simulation_config"]["t_max"] = "soon"
        response = client.post("/api/simulate", json={"blueprint": blueprint})
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertIn("must be a number", detail_text(response))

    def test_a_pde_grid_key_the_solver_ignores_is_refused_not_defaulted(self):
        """nx/ny are not fields solve_pde reads, so the grid fell back to 50x50."""
        blueprint = pde_blueprint()
        blueprint["spatial"].pop("x_grid")
        blueprint["spatial"].pop("y_grid")
        blueprint["spatial"]["nx"] = 0
        blueprint["spatial"]["ny"] = 0
        response = client.post("/api/simulate", json={"blueprint": blueprint})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("spatial.nx", detail)
        self.assertIn("x_grid", detail)
        self.assertIn("50x50", detail)

    def test_a_zero_pde_grid_dimension_is_refused(self):
        blueprint = pde_blueprint()
        blueprint["spatial"]["x_grid"] = 0
        blueprint["spatial"]["y_grid"] = 0
        response = client.post("/api/simulate", json={"blueprint": blueprint})
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertIn("spatial.x_grid", detail_text(response))

    def test_a_negative_pde_dt_is_refused_not_reported_as_solved(self):
        response = client.post("/api/simulate", json={"blueprint": pde_blueprint(dt=-0.1)})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("simulation_config.dt", detail)
        self.assertIn("-0.1", detail)

    def test_a_negative_pde_dt_no_longer_returns_a_single_frame_as_success(self):
        response = client.post("/api/simulate", json={"blueprint": pde_blueprint(dt=-0.1)})
        self.assertNotEqual(response.status_code, 200)
        self.assertNotIn("species", response.json())

    def test_a_negative_pde_horizon_is_refused(self):
        response = client.post("/api/simulate",
                               json={"blueprint": pde_blueprint(t_max=-10)})
        self.assertEqual(response.status_code, 422, response.text[:300])

    def test_solve1d_with_a_negative_dt_is_refused(self):
        response = solve_1d(one_d_field(), dt=-1.0)
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertIn("dt", detail_text(response))

    def test_solve1d_never_claims_to_have_solved_a_window_it_skipped(self):
        """steps=0 with a summary reading "Solved from t=0 to t=10" was the defect."""
        response = solve_1d(one_d_field(), dt=-1.0)
        self.assertNotEqual(response.status_code, 200)
        self.assertNotIn("summary", response.json())

    def test_a_negative_num_mcs_is_refused_not_answered_with_one_frame(self):
        response = client.post("/api/abm/simulate",
                               json={"blueprint": abm_blueprint(num_mcs=-5)})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("num_mcs", detail)
        self.assertIn("-5", detail)

    def test_a_negative_num_mcs_no_longer_returns_t_zero_only(self):
        response = client.post("/api/abm/simulate",
                               json={"blueprint": abm_blueprint(num_mcs=-5)})
        self.assertNotEqual(response.status_code, 200)
        self.assertNotIn("t", response.json())

    def test_a_negative_multiscale_num_mcs_is_refused(self):
        response = client.post("/api/multiscale/simulate", json={
            "abm_blueprint": abm_blueprint(), "num_mcs": -5, "save_every": 1})
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertIn("num_mcs", detail_text(response))

    def test_a_non_positive_save_every_is_refused(self):
        response = client.post("/api/abm/simulate",
                               json={"blueprint": abm_blueprint(save_every=0)})
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertIn("save_every", detail_text(response))

    def test_a_non_positive_lattice_dimension_is_refused(self):
        blueprint = abm_blueprint()
        blueprint["grid"] = {"width": 0, "height": 16}
        response = client.post("/api/abm/simulate", json={"blueprint": blueprint})
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertIn("grid.width", detail_text(response))

    def test_a_valid_pde_run_is_still_accepted(self):
        response = client.post("/api/simulate", json={"blueprint": pde_blueprint()})
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertGreater(len(response.json()["t"]), 1)


# =========================================================================
# P7 -- an unrecognised target scored as met
# =========================================================================

class TargetEvaluationHonestyTests(unittest.TestCase):
    """Measured before: targets [{"type": "not_a_target_type", "species": "ZZZ"}]
    returned met_count 1 of 1 with detail "Unknown metric." Anything driving a
    refinement loop off met_count concludes a requirement was satisfied that was
    never evaluated. A known metric on an unknown species was graded met=false with
    detail "No simulation data", which never names the species.
    """

    UNKNOWN_METRIC = [{"type": "not_a_target_type", "species": "ZZZ"}]

    def test_an_unknown_metric_is_never_counted_as_met(self):
        response = client.post("/api/evaluate", json={
            "blueprint": hill_blueprint(), "targets": self.UNKNOWN_METRIC})
        # The species is unknown too, so the request is refused before grading.
        self.assertEqual(response.status_code, 422, response.text[:300])
        self.assertIn("ZZZ", detail_text(response))

    def test_an_unknown_metric_on_a_real_species_is_refused_not_met(self):
        response = client.post("/api/evaluate", json={
            "blueprint": hill_blueprint(),
            "targets": [{"type": "not_a_target_type", "species": "A"}]})
        self.assertEqual(response.status_code, 200, response.text[:300])
        payload = response.json()
        self.assertEqual(payload["met_count"], 0)
        self.assertFalse(payload["results"][0]["met"])
        self.assertTrue(payload["results"][0]["refused"])
        self.assertEqual(payload["refused_count"], 1)

    def test_an_unknown_species_is_named_not_graded(self):
        response = client.post("/api/evaluate", json={
            "blueprint": hill_blueprint(),
            "targets": [{"type": "steady_state", "species": "ZZZ",
                         "value": 1.0, "tolerance": 0.1}]})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("ZZZ", detail)
        self.assertIn("A", detail)              # the species the model does have
        self.assertNotIn("No simulation data", detail)

    def test_a_pde_field_declared_only_in_spatial_reactions_is_recognised(self):
        """A PDE blueprint's species live under spatial.reactions, so grading "U"
        must not be mistaken for an unknown species."""
        response = client.post("/api/evaluate", json={
            "blueprint": pde_blueprint(),
            "targets": [{"type": "steady_state", "species": "U",
                         "value": 1.0, "tolerance": 10.0}]})
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertEqual(response.json()["graded_on"], "the spatial PDE field")

    def test_a_real_target_still_grades(self):
        response = client.post("/api/evaluate", json={
            "blueprint": hill_blueprint(),
            "targets": [{"type": "steady_state", "species": "A",
                         "value": 5.0, "tolerance": 10.0}]})
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertEqual(response.json()["total"], 1)


# =========================================================================
# P8 -- invalid parameters silently rewritten, or answered with a zero
# =========================================================================

class SilentRewriteTests(unittest.TestCase):
    """Measured before: /api/sample accepted method "telepathy" and silently used
    LHS; n_samples=-5 reported n_requested=1; param_bounds min=5.0 max=0.1 was
    silently swapped. /api/sensitivity returned {"k": 0.0} for target_species "NOPE"
    and 0.0 for an unknown parameter name -- and 0.0 is exactly what a genuinely
    insensitive parameter scores, so the answer could not be told from a real one.
    """

    BOUNDS = {"deg_A": {"min": 0.05, "max": 0.5}}

    def test_an_unimplemented_sampling_method_is_refused(self):
        response = client.post("/api/sample", json={
            "blueprint": hill_blueprint(), "param_bounds": self.BOUNDS,
            "n_samples": 4, "method": "telepathy"})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("telepathy", detail)
        self.assertIn("lhs", detail)            # the accepted set is listed

    def test_the_response_never_echoes_a_method_it_did_not_use(self):
        response = client.post("/api/sample", json={
            "blueprint": hill_blueprint(), "param_bounds": self.BOUNDS,
            "n_samples": 4, "method": "telepathy"})
        self.assertNotEqual(response.status_code, 200)

    def test_a_negative_sample_count_is_refused_not_clamped_to_one(self):
        response = client.post("/api/sample", json={
            "blueprint": hill_blueprint(), "param_bounds": self.BOUNDS,
            "n_samples": -5})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("n_samples", detail)
        self.assertIn("-5", detail)

    def test_inverted_bounds_are_refused_not_swapped(self):
        response = client.post("/api/sample", json={
            "blueprint": hill_blueprint(),
            "param_bounds": {"deg_A": {"min": 5.0, "max": 0.1}}, "n_samples": 4})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("min=5", detail)
        self.assertIn("max=0.1", detail)

    def test_a_sample_count_over_the_server_limit_is_refused(self):
        response = client.post("/api/sample", json={
            "blueprint": hill_blueprint(), "param_bounds": self.BOUNDS,
            "n_samples": main.MAX_SAMPLES + 1})
        self.assertEqual(response.status_code, 422, response.text[:300])

    def test_a_valid_sample_request_still_runs(self):
        response = client.post("/api/sample", json={
            "blueprint": hill_blueprint(), "param_bounds": self.BOUNDS,
            "n_samples": 4, "method": "lhs"})
        self.assertEqual(response.status_code, 200, response.text[:300])
        payload = response.json()
        self.assertEqual(payload["method"], "lhs")
        self.assertEqual(payload["n_requested"], 4)

    def test_sample_still_refuses_an_unknown_parameter_with_400(self):
        """The one case this route already got right must keep its status."""
        response = client.post("/api/sample", json={
            "blueprint": hill_blueprint(),
            "param_bounds": {"nope": {"min": 0.1, "max": 1.0}}, "n_samples": 4})
        self.assertEqual(response.status_code, 400, response.text[:300])
        self.assertIn("None of the requested parameters exist",
                      detail_text(response))

    def test_sensitivity_agrees_with_sample_on_an_unknown_parameter(self):
        response = client.post("/api/sensitivity", json={
            "blueprint": hill_blueprint(), "target_species": "A",
            "param_names": ["not_a_param"]})
        self.assertEqual(response.status_code, 400, response.text[:300])
        self.assertIn("None of the requested parameters exist",
                      detail_text(response))

    def test_sensitivity_never_reports_a_fabricated_zero_for_an_unknown_parameter(self):
        response = client.post("/api/sensitivity", json={
            "blueprint": hill_blueprint(), "target_species": "A",
            "param_names": ["not_a_param"]})
        self.assertNotIn("sensitivities", response.json())

    def test_sensitivity_names_a_partially_unknown_parameter_list(self):
        response = client.post("/api/sensitivity", json={
            "blueprint": hill_blueprint(), "target_species": "A",
            "param_names": ["deg_A", "not_a_param"]})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("not_a_param", detail)
        self.assertIn("0.0", detail)            # says what it would have reported

    def test_sensitivity_refuses_an_unknown_target_species(self):
        response = client.post("/api/sensitivity", json={
            "blueprint": hill_blueprint(), "target_species": "NOPE",
            "param_names": ["deg_A"]})
        self.assertEqual(response.status_code, 422, response.text[:300])
        detail = detail_text(response)
        self.assertIn("NOPE", detail)
        self.assertIn("A", detail)

    def test_sensitivity_for_an_unknown_species_returns_no_numbers_at_all(self):
        response = client.post("/api/sensitivity", json={
            "blueprint": hill_blueprint(), "target_species": "NOPE",
            "param_names": ["deg_A"]})
        self.assertNotIn("sensitivities", response.json())

    def test_a_valid_sensitivity_request_still_runs(self):
        response = client.post("/api/sensitivity", json={
            "blueprint": hill_blueprint(), "target_species": "A",
            "param_names": ["deg_A"]})
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertIn("deg_A", response.json()["sensitivities"])
        self.assertNotEqual(response.json()["sensitivities"]["deg_A"], 0.0)


# =========================================================================
# P12 -- resuming a run that was never paused
# =========================================================================

class ResumeStateTests(unittest.TestCase):
    """Measured before: POST /api/runs/{id}/resume returned 200 for a run that was
    never paused, moving it to `running` with no worker thread behind it.
    """

    def _ready_run(self):
        project = client.post("/api/project/new", json={
            "domain": geo.make_rectangle(24.0, 24.0)}).json()
        project["selected_approach"] = "mpc"
        project["approaches"] = {"mpc": {"target": 1.0, "duration": 6.0,
                                         "control_interval": 0.5,
                                         "prediction_horizon": 8,
                                         "control_horizon": 2}}
        created = client.post("/api/runs", json={"project": project, "start": False})
        self.assertEqual(created.status_code, 200, created.text[:300])
        return created.json()["run_id"]

    def test_resuming_a_never_paused_run_is_a_conflict(self):
        run_id = self._ready_run()
        response = client.post(f"/api/runs/{run_id}/resume")
        self.assertEqual(response.status_code, 409, response.text[:300])
        self.assertIn("not 'paused'", detail_text(response))

    def test_the_run_is_not_left_in_a_phantom_running_state(self):
        run_id = self._ready_run()
        client.post(f"/api/runs/{run_id}/resume")
        self.assertEqual(client.get(f"/api/runs/{run_id}").json()["state"], "ready")

    def test_resuming_an_unknown_run_is_still_404(self):
        self.assertEqual(client.post("/api/runs/run_missing/resume").status_code, 404)


# =========================================================================
# Cross-cutting: no hardened route may answer with the generic 500 body
# =========================================================================

class NoGenericFiveHundredTests(unittest.TestCase):
    """The global handler's body -- {"error": "Internal server error"} -- is the
    signal that an endpoint let an exception escape. None of the payloads above may
    produce it, whatever else they produce.
    """

    CASES = [
        ("/api/abm/simulate", {"blueprint": {}}),
        ("/api/multiscale/simulate", {"abm_blueprint": {}}),
        ("/api/topology/cycles", {"topology": {"nodes": "x", "edges": "y", "cells": []}}),
        ("/api/topology/cycles", {"topology": {"nodes": "x", "edges": ["e1"], "cells": []}}),
        ("/api/topology/export",
         {"topology": {"nodes": ["n1"], "edges": ["e1"], "cells": ["c1"]}}),
        ("/api/compile", {"blueprint": {"nodes": [{"initial_value": 1.0}], "edges": []}}),
        ("/api/simulate", {"blueprint": {"nodes": [{"initial_value": 1.0}], "edges": []}}),
        ("/api/simulate", {"blueprint": {
            "nodes": [{"id": "X", "initial_value": 1.0}], "edges": [],
            "odes": {"X": "1/0"}, "simulation_config": {"t_max": 5.0}}}),
        ("/api/sensitivity", {"blueprint": hill_blueprint(), "target_species": "NOPE",
                              "param_names": ["deg_A"]}),
    ]

    def test_no_reported_payload_reaches_the_global_handler(self):
        for path, body in self.CASES:
            response = client.post(path, json=body)
            self.assertNotEqual(response.status_code, 500,
                                msg=f"{path} still 500s on {body}")
            self.assertNotIn("Internal server error", response.text,
                             msg=f"{path} returned the generic body")

    def test_every_refusal_carries_a_message_longer_than_a_python_repr(self):
        """{"detail": "'id'"} was 4 characters. A refusal must be actionable."""
        for path, body in self.CASES:
            response = client.post(path, json=body)
            self.assertGreater(len(detail_text(response)), 25,
                               msg=f"{path} refused with an unhelpful detail")


if __name__ == "__main__":
    unittest.main()
