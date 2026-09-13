"""Tests for PDE model configuration, expression safety and compilation."""

import unittest

import numpy as np

import geometry as geo
import pde_model as pm


def _codes(issues):
    return {i.code for i in issues}


def _model(**overrides):
    model = {
        "parameters": {"D": 0.5, "k": 0.2},
        "fields": [dict(pm.FIELD_DEFAULTS, name="u", units="mM", diffusion="D",
                        initial="1.0", reaction="-k*u", source="0",
                        t_start=0.0, t_end=5.0, output_interval=0.5)],
    }
    model.update(overrides)
    return model


class ExpressionSafetyTests(unittest.TestCase):
    def test_valid_expression_parses(self):
        expression = pm.parse_safe("D*u + exp(-x)", ["D", "u", "x"])
        self.assertEqual(sorted(str(s) for s in expression.free_symbols), ["D", "u", "x"])

    def test_unknown_symbol_is_named_in_the_error(self):
        with self.assertRaises(pm.ExpressionError) as raised:
            pm.parse_safe("k*u", ["u"])
        message = str(raised.exception)
        self.assertIn("k", message)
        self.assertIn("Available", message)

    def test_dunder_is_refused(self):
        with self.assertRaises(pm.ExpressionError):
            pm.parse_safe("u.__class__", ["u"])

    def test_import_attempt_is_refused(self):
        for hostile in ("__import__('os')", "open('x')", "eval('1')", "exec('pass')"):
            with self.assertRaises(pm.ExpressionError, msg=hostile):
                pm.parse_safe(hostile, ["u"])

    def test_disallowed_function_is_refused(self):
        # 'gamma' is a real sympy function but is not in the allow-list, so it must
        # surface as an undefined name rather than silently resolving.
        with self.assertRaises(pm.ExpressionError):
            pm.parse_safe("gamma(u)", ["u"])

    def test_allowed_functions_work(self):
        for text in ("exp(u)", "sqrt(u)", "tanh(u)", "Max(u, 0)", "abs(u)"):
            pm.parse_safe(text, ["u"])

    def test_empty_expression_becomes_zero(self):
        self.assertEqual(pm.parse_safe("", ["u"]), 0)
        self.assertEqual(pm.parse_safe(None, ["u"]), 0)

    def test_lambdified_constant_broadcasts(self):
        func = pm.lambdify_scalar(pm.parse_safe("2.5", ["x"]), ["x"])
        out = func(np.zeros(4))
        self.assertIsInstance(out, np.ndarray)
        np.testing.assert_allclose(out, 2.5)

    def test_lambdified_expression_evaluates_elementwise(self):
        func = pm.lambdify_scalar(pm.parse_safe("x**2", ["x"]), ["x"])
        np.testing.assert_allclose(func(np.array([1.0, 2.0, 3.0])), [1.0, 4.0, 9.0])


class PresetTests(unittest.TestCase):
    def test_every_preset_is_valid(self):
        for name in pm.preset_names():
            preset = pm.get_preset(name)
            issues = pm.validate_model(preset, geo.make_interval(1.0))
            self.assertEqual([i for i in issues if i.severity == "error"], [],
                             msg=f"preset {name} should validate")

    def test_every_preset_compiles(self):
        for name in pm.preset_names():
            preset = pm.get_preset(name)
            compiled = pm.compile_field(preset, preset["fields"][0]["name"],
                                        geo.make_interval(1.0))
            self.assertIn("reaction", compiled)
            self.assertTrue(callable(compiled["reaction"]))

    def test_unknown_preset_lists_the_options(self):
        with self.assertRaises(ValueError) as raised:
            pm.get_preset("teleportation")
        self.assertIn("diffusion", str(raised.exception))

    def test_advection_preset_has_a_velocity(self):
        compiled = pm.compile_field(pm.get_preset("advection_diffusion"), "u",
                                    geo.make_interval(1.0))
        self.assertAlmostEqual(compiled["advection"]["vx"], 1.0)


class ValidationTests(unittest.TestCase):
    def test_valid_model_passes(self):
        self.assertEqual([i for i in pm.validate_model(_model()) if i.severity == "error"], [])

    def test_no_fields_is_an_error(self):
        self.assertIn("no_fields", _codes(pm.validate_model({"fields": []})))

    def test_duplicate_field_names_rejected(self):
        model = _model()
        model["fields"].append(dict(model["fields"][0]))
        self.assertIn("field_name_duplicate", _codes(pm.validate_model(model)))

    def test_reserved_coordinate_name_rejected(self):
        model = _model()
        model["fields"][0]["name"] = "x"
        self.assertIn("field_name_reserved", _codes(pm.validate_model(model)))

    def test_missing_units_is_a_warning_not_an_error(self):
        model = _model()
        model["fields"][0]["units"] = ""
        issues = pm.validate_model(model)
        self.assertIn("field_units_missing", _codes(issues))
        self.assertEqual([i for i in issues if i.severity == "error"], [])

    def test_undefined_parameter_in_reaction_rejected(self):
        model = _model()
        model["fields"][0]["reaction"] = "-lambda_decay*u"
        self.assertIn("field_reaction_invalid", _codes(pm.validate_model(model)))

    def test_negative_diffusion_rejected(self):
        model = _model(parameters={"D": -1.0, "k": 0.1})
        self.assertIn("field_diffusion_negative", _codes(pm.validate_model(model)))

    def test_solution_dependent_diffusion_rejected(self):
        model = _model()
        model["fields"][0]["diffusion"] = "D*u"
        self.assertIn("field_diffusion_not_constant", _codes(pm.validate_model(model)))

    def test_time_dependent_diffusion_rejected(self):
        model = _model()
        model["fields"][0]["diffusion"] = "D*t"
        self.assertIn("field_diffusion_not_constant", _codes(pm.validate_model(model)))

    def test_space_dependent_advection_rejected(self):
        model = _model()
        model["fields"][0]["advection"] = {"vx": "x", "vy": 0.0}
        self.assertIn("field_advection_not_constant", _codes(pm.validate_model(model)))

    def test_inverted_time_range_rejected(self):
        model = _model()
        model["fields"][0]["t_end"] = -1.0
        self.assertIn("field_time_range_invalid", _codes(pm.validate_model(model)))

    def test_non_positive_output_interval_rejected(self):
        model = _model()
        model["fields"][0]["output_interval"] = 0.0
        self.assertIn("field_output_interval_invalid", _codes(pm.validate_model(model)))

    def test_coarse_output_interval_warns(self):
        model = _model()
        model["fields"][0]["output_interval"] = 99.0
        self.assertIn("field_output_interval_coarse", _codes(pm.validate_model(model)))

    def test_bad_parameter_value_rejected(self):
        self.assertIn("parameter_value_invalid",
                      _codes(pm.validate_model(_model(parameters={"D": "abc"}))))

    def test_y_is_not_available_in_a_1d_domain(self):
        model = _model()
        model["fields"][0]["source"] = "y"
        issues = pm.validate_model(model, geo.make_interval(1.0))
        self.assertIn("field_source_invalid", _codes(issues))
        # ... but it IS available in 2D.
        self.assertEqual(
            [i for i in pm.validate_model(model, geo.make_rectangle(1.0, 1.0))
             if i.severity == "error"], [])


class CompilationTests(unittest.TestCase):
    def test_compile_substitutes_parameters(self):
        compiled = pm.compile_field(_model(), "u")
        self.assertAlmostEqual(compiled["diffusion"], 0.5)
        # reaction = -k*u with k=0.2
        values = compiled["reaction"](np.array([1.0, 2.0]), np.zeros(2), np.zeros(2), 0.0)
        np.testing.assert_allclose(values, [-0.2, -0.4])

    def test_compile_refuses_an_invalid_model(self):
        model = _model()
        model["fields"][0]["reaction"] = "-missing*u"
        with self.assertRaises(ValueError):
            pm.compile_field(model, "u")

    def test_unknown_field_name_raises_with_the_list(self):
        with self.assertRaises(KeyError) as raised:
            pm.compile_field(_model(), "ghost")
        self.assertIn("u", str(raised.exception))

    def test_initial_expression_evaluates_over_space(self):
        model = _model()
        model["fields"][0]["initial"] = "exp(-x**2)"
        compiled = pm.compile_field(model, "u")
        x = np.array([0.0, 1.0])
        np.testing.assert_allclose(compiled["initial"](x, np.zeros_like(x)),
                                   [1.0, np.exp(-1.0)])

    def test_source_expression_evaluates_over_space_and_time(self):
        model = _model()
        model["fields"][0]["source"] = "x*t"
        compiled = pm.compile_field(model, "u")
        x = np.array([1.0, 2.0])
        np.testing.assert_allclose(compiled["source"](x, np.zeros_like(x), 3.0), [3.0, 6.0])


class SummaryTests(unittest.TestCase):
    def test_latex_contains_the_field_and_the_reaction(self):
        latex = pm.equation_latex(_model()["fields"][0], {"D": 0.5, "k": 0.2})
        self.assertIn(r"\partial u", latex)
        self.assertIn(r"\nabla^2 u", latex)

    def test_plain_summary_mentions_units_and_time_range(self):
        summary = pm.plain_language_summary(_model()["fields"][0], {"D": 0.5, "k": 0.2})
        self.assertIn("mM", summary)
        self.assertIn("t=0", summary)
        self.assertIn("t=5", summary)
        self.assertIn("diffusion", summary)

    def test_advection_is_mentioned_when_present(self):
        preset = pm.get_preset("advection_diffusion")
        summary = pm.plain_language_summary(preset["fields"][0], preset["parameters"])
        self.assertIn("carried by flow", summary)

    def test_pure_diffusion_summary_has_no_reaction_clause(self):
        preset = pm.get_preset("diffusion")
        summary = pm.plain_language_summary(preset["fields"][0], preset["parameters"])
        self.assertNotIn("changes locally", summary)


if __name__ == "__main__":
    unittest.main()
