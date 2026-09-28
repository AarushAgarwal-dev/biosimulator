"""Offline tests for sbml_import (exact SBML -> blueprint translation).

The network benchmark tools/verify_sbml_import.py compares 8 curated BioModels entries against
libroadrunner (all match within 1e-4). These tests pin the translation rules on small documents
so they run anywhere.
"""
import math
import unittest

import numpy as np

import sbml_import
from simulation_engine import ODEModel

L2 = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level2/version4" level="2" version="4">
 <model id="m" name="test model">
  <listOfFunctionDefinitions>
   <functionDefinition id="mm">
    <math xmlns="http://www.w3.org/1998/Math/MathML"><lambda>
      <bvar><ci>V</ci></bvar><bvar><ci>S</ci></bvar><bvar><ci>K</ci></bvar>
      <apply><divide/><apply><times/><ci>V</ci><ci>S</ci></apply><apply><plus/><ci>K</ci><ci>S</ci></apply></apply>
    </lambda></math>
   </functionDefinition>
  </listOfFunctionDefinitions>
  <listOfCompartments><compartment id="cell" size="2"/></listOfCompartments>
  <listOfSpecies>
   <species id="A" compartment="cell" initialConcentration="10"/>
   <species id="B" compartment="cell" initialAmount="0"/>
   <species id="E" compartment="cell" initialConcentration="1" boundaryCondition="true"/>
  </listOfSpecies>
  <listOfParameters><parameter id="Vmax" value="3"/><parameter id="lambda" value="0.5"/></listOfParameters>
  <listOfReactions>
   <reaction id="conv" reversible="false">
    <listOfReactants><speciesReference species="A" stoichiometry="2"/></listOfReactants>
    <listOfProducts><speciesReference species="B"/></listOfProducts>
    <listOfModifiers><modifierSpeciesReference species="E"/></listOfModifiers>
    <kineticLaw>
     <math xmlns="http://www.w3.org/1998/Math/MathML">
      <apply><times/><ci>cell</ci><ci>E</ci><apply><ci>mm</ci><ci>Vmax</ci><ci>A</ci><ci>Km</ci></apply></apply>
     </math>
     <listOfParameters><parameter id="Km" value="4"/></listOfParameters>
    </kineticLaw>
   </reaction>
   <reaction id="decay" reversible="false">
    <listOfReactants><speciesReference species="B"/></listOfReactants>
    <kineticLaw><math xmlns="http://www.w3.org/1998/Math/MathML">
     <apply><times/><ci>cell</ci><ci>lambda</ci><ci>B</ci></apply></math></kineticLaw>
   </reaction>
  </listOfReactions>
 </model>
</sbml>"""


class ExactTranslationTests(unittest.TestCase):
    def setUp(self):
        self.bp = sbml_import.sbml_to_blueprint(L2)

    def test_species_parameters_and_reserved_names(self):
        ids = {n["id"] for n in self.bp["nodes"]}
        self.assertEqual(ids, {"A", "B"})                      # boundary species E is a parameter
        # 'E' is SymPy's Euler constant, so the SBML id E is renamed (and the rename disclosed).
        self.assertEqual(self.bp["parameters"]["E_sb"], 1.0)
        self.assertEqual(self.bp["_sbml_report"]["renamed"].get("E"), "E_sb")
        self.assertEqual(self.bp["parameters"]["cell"], 2.0)
        self.assertIn("lambda_sb", self.bp["parameters"])       # 'lambda' is a Python keyword
        self.assertIn("conv__Km", self.bp["parameters"])        # local parameter is scoped
        self.assertEqual(self.bp["_compiler"], "sbml_import/exact")

    def test_dynamics_match_the_hand_derived_equations(self):
        # d[A]/dt = -2 * cell*E*Vmax*A/(Km+A) / cell ; d[B]/dt = (cell*E*Vmax*A/(Km+A) - cell*lambda*B)/cell
        r = ODEModel(self.bp).simulate(5.0, num_points=51)
        from scipy.integrate import solve_ivp

        def f(t, y):
            a, b = y
            v = 1.0 * 3.0 * a / (4.0 + a)
            return [-2 * v, v - 0.5 * b]
        ref = solve_ivp(f, (0, 5), [10.0, 0.0], t_eval=np.linspace(0, 5, 51), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(r["species"]["A"], ref.y[0], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(r["species"]["B"], ref.y[1], rtol=1e-5, atol=1e-7)

    def test_level2_default_time_unit_sets_the_horizon(self):
        self.assertEqual(self.bp["simulation_config"]["t_max"], 3600.0)
        self.assertIn("second", self.bp["_sbml_report"]["time_unit"])

    def test_events_are_reported_not_dropped_silently(self):
        doc = L2.replace("</listOfReactions>", "</listOfReactions><listOfEvents><event id='e1'/></listOfEvents>")
        bp = sbml_import.sbml_to_blueprint(doc)
        self.assertIn("event", bp["_llm_notice"])
        self.assertTrue(any("event" in u for u in bp["_sbml_report"]["unsupported"]))

    def test_not_sbml_is_rejected(self):
        with self.assertRaises(sbml_import.SBMLImportError):
            sbml_import.sbml_to_blueprint("<html><body>not sbml</body></html>")
        with self.assertRaises(sbml_import.SBMLImportError):
            sbml_import.sbml_to_blueprint("not xml at all <")


class AssignmentAndRateRuleTests(unittest.TestCase):
    def test_rules(self):
        doc = """<sbml xmlns="http://www.sbml.org/sbml/level3/version1/core" level="3" version="1">
         <model id="r" timeUnits="minute">
          <listOfCompartments><compartment id="c" size="1" constant="true"/></listOfCompartments>
          <listOfSpecies><species id="X" compartment="c" initialConcentration="1" hasOnlySubstanceUnits="false"
             boundaryCondition="false" constant="false"/></listOfSpecies>
          <listOfParameters><parameter id="k" value="0.2" constant="true"/>
            <parameter id="Y" value="0" constant="false"/><parameter id="Z" value="0" constant="false"/></listOfParameters>
          <listOfRules>
           <assignmentRule variable="Y"><math xmlns="http://www.w3.org/1998/Math/MathML">
             <apply><times/><cn>2</cn><ci>X</ci></apply></math></assignmentRule>
           <rateRule variable="Z"><math xmlns="http://www.w3.org/1998/Math/MathML"><ci>Y</ci></math></rateRule>
           <rateRule variable="X"><math xmlns="http://www.w3.org/1998/Math/MathML">
             <apply><times/><apply><minus/><ci>k</ci></apply><ci>X</ci></apply></math></rateRule>
          </listOfRules>
         </model></sbml>"""
        bp = sbml_import.sbml_to_blueprint(doc)
        self.assertEqual(bp["simulation_config"]["t_max"], 300.0)            # minutes
        r = ODEModel(bp).simulate(10.0, num_points=11)
        x_exact = math.exp(-0.2 * 10.0)
        self.assertAlmostEqual(r["species"]["X"][-1], x_exact, places=6)
        # Z = integral of 2X = 10*(1 - exp(-0.2 t))
        self.assertAlmostEqual(r["species"]["Z"][-1], 10.0 * (1 - x_exact), places=5)


if __name__ == "__main__":
    unittest.main()
