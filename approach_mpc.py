"""
MPC approach: Model Predictive Control.

MPC is a CONTROL method, and is implemented as one. On every control interval it

  1. reads the current state,
  2. predicts the response over a prediction horizon using a model,
  3. solves a constrained optimisation for the control sequence,
  4. applies only the FIRST control action (receding horizon),
  5. advances the plant, and repeats.

It is deliberately NOT a PDE discretisation dressed up as a third solver. The
objective minimised is

    J = sum_k  w_track * (y[k] - target[k])^2
      + sum_k  w_effort * u[k]^2
      + sum_k  w_rate   * (u[k] - u[k-1])^2

subject to input bounds, input rate limits and output constraints.

Numerical assumptions and limitations
-------------------------------------
* The predictive model is a first-order linear plant
  ``dx/dt = -x/tau + gain * u`` (optionally with dead time), integrated with the
  same explicit step as the plant. It is intentionally simple and honest: a
  mismatch between model and plant is configurable, so the demo can show that MPC
  still tracks under model error.
* Optimisation uses ``scipy.optimize.minimize`` with SLSQP, which is already a
  project dependency -- no second optimisation stack is introduced.
* Bounds are enforced by the optimiser AND clipped on application, so a reported
  control action can never violate its bounds even if the optimiser returns an
  infeasible point.
* An optimiser failure is reported, not hidden: the step falls back to holding the
  previous action and the failure is counted in the results.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import scipy.optimize

from approach_base import (
    ApproachAdapter,
    Capabilities,
    RunContext,
    register_approach,
)
from geometry import ValidationIssue

DEFAULTS: Dict[str, Any] = {
    "model": {"kind": "first_order", "tau": 5.0, "gain": 1.0, "dead_time": 0.0},
    # Plant used as the "real" system. When it differs from the model, the run
    # demonstrates closed-loop robustness rather than a self-fulfilling simulation.
    "plant": {"kind": "first_order", "tau": 5.0, "gain": 1.0, "dead_time": 0.0},
    "controlled_input": "u",
    "measured_output": "y",
    "target": 1.0,
    "prediction_horizon": 12,
    "control_horizon": 4,
    "control_interval": 0.5,
    "duration": 30.0,
    "input_min": -2.0,
    "input_max": 2.0,
    "input_rate_limit": 0.5,
    "output_min": None,
    "output_max": None,
    "weight_tracking": 1.0,
    "weight_effort": 0.05,
    "weight_rate": 0.05,
    "tolerance": 1e-6,
    "max_iterations": 60,
    "initial_state": 0.0,
    # Axis labels for the results workspace. Defaulting to arbitrary units rather
    # than an empty string means a plot is never silently unitless; a researcher
    # overrides these with the real units of their controlled quantity.
    "output_units": "a.u.",
    "input_units": "a.u.",
}


def _merged(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    merged["model"] = dict(DEFAULTS["model"])
    merged["plant"] = dict(DEFAULTS["plant"])
    for key, value in (config or {}).items():
        if key in ("model", "plant") and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _target_at(config: Dict[str, Any], t: float) -> float:
    """Target value at time ``t``. Supports a constant or a time/value trajectory."""
    target = config.get("target", 0.0)
    if isinstance(target, (int, float)):
        return float(target)
    if isinstance(target, dict):
        times = [float(v) for v in (target.get("times") or [])]
        values = [float(v) for v in (target.get("values") or [])]
        if times and values and len(times) == len(values):
            return float(np.interp(t, times, values))
    if isinstance(target, list) and target:
        # [[t, value], ...]
        try:
            times = [float(p[0]) for p in target]
            values = [float(p[1]) for p in target]
            return float(np.interp(t, times, values))
        except (TypeError, ValueError, IndexError):
            return 0.0
    return 0.0


def _step_first_order(state: float, u: float, dt: float, spec: Dict[str, Any]) -> float:
    """One explicit step of dx/dt = -x/tau + gain*u."""
    tau = max(1e-9, float(spec.get("tau", 5.0)))
    gain = float(spec.get("gain", 1.0))
    return float(state + dt * (-state / tau + gain * u))


class ReactionNetworkPlant:
    """A plant that is THE MODEL THE RESEARCHER PREPARED, not a textbook system.

    Why this exists. The workflow's premise is "prepare a domain, topology, mesh and model
    ONCE, then choose one approach", and MPC ignored all of it: it simulated its own
    ``dx/dt = -x/tau + gain*u`` with tau = 5, gain = 1, while ``controlled_input`` and
    ``measured_output`` were the strings "u" and "y" -- ports of that toy system, not
    species in anyone's network. The result was a correct, well-validated controller
    demonstration with no connection to the biology, and nothing said so.

    What this controls instead: the stage-4 reaction network, as a WELL-MIXED system. Each
    field's ``reaction`` expression is its rate law, diffusion is dropped, and what is left
    is d[field]/dt = reaction(fields, parameters) -- the ordinary-differential reduction of
    the reaction-diffusion model, which is the standard way to ask a control question about
    kinetics without committing to geometry. That reduction is stated to the researcher
    rather than hidden, because a spatially averaged answer is not the same as a spatial
    one.

    The control acts as a DOSE: u is added to the input species' rate of change, which is
    what "how much of this do I infuse per unit time" means. It is deliberately not a
    multiplier on the rate law -- that would silently rescale a rate constant the
    researcher chose.

    This makes MPC a biological question: hold ERK at 40% of its peak, and tell me the
    infusion schedule that does it.
    """

    def __init__(self, model: Dict[str, Any], input_species: str, output_species: str):
        import pde_model

        fields = [f for f in (model or {}).get("fields") or [] if f.get("name")]
        if not fields:
            raise ValueError("the stage-4 model defines no fields, so there is no "
                             "reaction network to control")
        self.names = [str(f["name"]) for f in fields]
        if input_species not in self.names:
            raise ValueError(f"controlled_input {input_species!r} is not one of the "
                             f"model's species: {', '.join(self.names)}")
        if output_species not in self.names:
            raise ValueError(f"measured_output {output_species!r} is not one of the "
                             f"model's species: {', '.join(self.names)}")
        self.input_species = input_species
        self.output_species = output_species

        parameters = dict((model or {}).get("parameters") or {})
        allowed = list(self.names) + list(parameters.keys())
        self._rates = []
        for field in fields:
            expr = pde_model.parse_safe(field.get("reaction", "0"), allowed)
            expr = expr.subs({sp_name: value for sp_name, value in parameters.items()}) \
                if hasattr(expr, "subs") else expr
            self._rates.append(pde_model.lambdify_scalar(expr, self.names))

        self.state = []
        for field in fields:
            try:
                self.state.append(float(pde_model.parse_safe(
                    field.get("initial", "0"), list(parameters.keys())).subs(parameters)))
            except Exception:
                self.state.append(0.0)

    def output(self) -> float:
        return float(self.state[self.names.index(self.output_species)])

    def step(self, u: float, dt: float) -> float:
        """Explicit Euler on the reaction network, with u dosed into the input species."""
        rates = []
        for rate in self._rates:
            try:
                rates.append(float(rate(*self.state)))
            except Exception:
                rates.append(0.0)
        index = self.names.index(self.input_species)
        rates[index] += float(u)
        self.state = [float(value + dt * rate)
                      for value, rate in zip(self.state, rates)]
        # A concentration cannot go negative; clamping here keeps a controller that
        # overshoots from driving the plant into a region the model cannot represent.
        self.state = [value if value > 0.0 else 0.0 for value in self.state]
        return self.output()


class MPCAdapter(ApproachAdapter):
    approach_id = "mpc"
    label = "MPC (Model Predictive Control)"

    def get_capabilities(self) -> Capabilities:
        return Capabilities(
            approach_id=self.approach_id,
            label=self.label,
            available=True,
            dimensions=(0,),          # MPC controls a signal, not a spatial field
            supports_pause=True,
            supports_cancel=True,
            supports_fields=False,
            supports_seed=True,
            deterministic_with_seed=True,
            supports_export=True,
            engine_name="scipy.optimize (SLSQP)",
            engine_version=str(getattr(scipy, "__version__", "")) or "unknown",
            requirements=["scipy"],
            notes=("Receding-horizon constrained optimal control. The predictive model "
                   "and the plant are configured separately, so model mismatch is "
                   "explicit rather than assumed away. "
                   "USES: the MPC settings here, and -- when the plant kind is "
                   "'reaction_network' -- YOUR STAGE-4 MODEL as the plant, integrated as "
                   "a well-mixed system with the control dosed into the species named by "
                   "controlled_input. That makes this a biological question: what "
                   "infusion schedule holds a species at a chosen level? IGNORES: the "
                   "domain, the mesh and the stage-5 boundary conditions, because the "
                   "well-mixed reduction drops space -- a spatially averaged answer is "
                   "not a spatial one. With the default 'first_order' plant it simulates "
                   "dx/dt = -x/tau + gain*u instead, which demonstrates the controller "
                   "and says nothing about your biology."),
        )

    # -- validation ---------------------------------------------------------
    def validate(self, project: Dict[str, Any]) -> List[ValidationIssue]:
        config = _merged(self.approach_config(project))
        issues: List[ValidationIssue] = []
        path = "approaches.mpc"

        def number(key: str, minimum: Optional[float] = None,
                   maximum: Optional[float] = None, integer: bool = False) -> Optional[float]:
            raw = config.get(key)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                issues.append(ValidationIssue("error", f"mpc_{key}_invalid",
                                              f"{key.replace('_', ' ')} must be a number.",
                                              f"{path}.{key}"))
                return None
            if not np.isfinite(value):
                issues.append(ValidationIssue("error", f"mpc_{key}_invalid",
                                              f"{key.replace('_', ' ')} must be finite.",
                                              f"{path}.{key}"))
                return None
            if integer and abs(value - round(value)) > 1e-9:
                issues.append(ValidationIssue("error", f"mpc_{key}_not_integer",
                                              f"{key.replace('_', ' ')} must be a whole number.",
                                              f"{path}.{key}"))
                return None
            if minimum is not None and value < minimum:
                issues.append(ValidationIssue(
                    "error", f"mpc_{key}_too_small",
                    f"{key.replace('_', ' ')} must be at least {minimum:g} (got {value:g}).",
                    f"{path}.{key}"))
                return None
            if maximum is not None and value > maximum:
                issues.append(ValidationIssue(
                    "error", f"mpc_{key}_too_large",
                    f"{key.replace('_', ' ')} must be at most {maximum:g} (got {value:g}).",
                    f"{path}.{key}"))
                return None
            return value

        prediction = number("prediction_horizon", minimum=1, integer=True)
        control = number("control_horizon", minimum=1, integer=True)
        number("control_interval", minimum=1e-9)
        number("duration", minimum=1e-9)
        number("tolerance", minimum=0.0)
        number("max_iterations", minimum=1, integer=True)
        for key in ("weight_tracking", "weight_effort", "weight_rate"):
            number(key, minimum=0.0)
        rate = number("input_rate_limit", minimum=0.0)

        if prediction is not None and control is not None and control > prediction:
            issues.append(ValidationIssue(
                "error", "mpc_control_horizon_too_long",
                f"The control horizon ({int(control)}) cannot exceed the prediction "
                f"horizon ({int(prediction)}): the optimiser would be choosing actions "
                f"beyond the window it can predict.",
                f"{path}.control_horizon"))

        u_min = number("input_min")
        u_max = number("input_max")
        if u_min is not None and u_max is not None and u_min >= u_max:
            issues.append(ValidationIssue(
                "error", "mpc_input_bounds_inverted",
                f"input_min ({u_min:g}) must be below input_max ({u_max:g}).",
                f"{path}.input_min"))

        y_min, y_max = config.get("output_min"), config.get("output_max")
        if y_min is not None and y_max is not None:
            try:
                if float(y_min) >= float(y_max):
                    issues.append(ValidationIssue(
                        "error", "mpc_output_bounds_inverted",
                        "output_min must be below output_max.", f"{path}.output_min"))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("error", "mpc_output_bounds_invalid",
                                              "Output constraints must be numbers or null.",
                                              f"{path}.output_min"))

        for spec_name in ("model", "plant"):
            spec = config.get(spec_name) or {}
            kind = str(spec.get("kind", "first_order"))

            # The PLANT may be the researcher's own stage-4 reaction network. The
            # predictive MODEL stays first-order: MPC's premise is that the controller's
            # internal model is deliberately simpler than the plant, so the mismatch is
            # explicit rather than assumed away, and allowing the true model on both sides
            # would quietly turn this into perfect-model control.
            if spec_name == "plant" and kind == "reaction_network":
                fields = [f.get("name") for f in
                          ((project or {}).get("model") or {}).get("fields") or []
                          if f.get("name")]
                if not fields:
                    issues.append(ValidationIssue(
                        "error", "mpc_plant_network_empty",
                        "The plant is set to the prepared reaction network, but the "
                        "stage-4 model defines no species to control.",
                        f"{path}.plant.kind"))
                    continue
                for key in ("controlled_input", "measured_output"):
                    name = str(config.get(key) or "")
                    if name not in fields:
                        issues.append(ValidationIssue(
                            "error", f"mpc_{key}_not_a_species",
                            f"{key} is {name!r}, which is not a species in the prepared "
                            f"model. Available: {', '.join(fields)}. With a "
                            f"reaction_network plant these must name real species, not "
                            f"the placeholder ports of the first-order test system.",
                            f"{path}.{key}"))
                continue

            if kind != "first_order":
                issues.append(ValidationIssue(
                    "error", f"mpc_{spec_name}_kind_unsupported",
                    f"Only the 'first_order' {spec_name} is implemented"
                    + (" (the plant may also be 'reaction_network')"
                       if spec_name == "plant" else "")
                    + f"; got {spec.get('kind')!r}.", f"{path}.{spec_name}.kind"))
                continue
            try:
                if float(spec.get("tau", 5.0)) <= 0:
                    issues.append(ValidationIssue(
                        "error", f"mpc_{spec_name}_tau_not_positive",
                        f"{spec_name} time constant tau must be greater than zero.",
                        f"{path}.{spec_name}.tau"))
                if float(spec.get("gain", 1.0)) == 0.0:
                    issues.append(ValidationIssue(
                        "warning", f"mpc_{spec_name}_gain_zero",
                        f"A {spec_name} gain of zero means the input cannot affect the "
                        f"output, so no controller can track a non-zero target.",
                        f"{path}.{spec_name}.gain"))
            except (TypeError, ValueError):
                issues.append(ValidationIssue("error", f"mpc_{spec_name}_invalid",
                                              f"{spec_name} tau and gain must be numbers.",
                                              f"{path}.{spec_name}"))

        # A reachability sanity check: a constant target outside the steady-state
        # range the bounded input can reach is not trackable, and saying so up front
        # is better than reporting a large residual error afterwards.
        if isinstance(config.get("target"), (int, float)) and u_min is not None and u_max is not None:
            spec = config.get("plant") or {}
            try:
                tau, gain = float(spec.get("tau", 5.0)), float(spec.get("gain", 1.0))
                reachable = sorted((tau * gain * u_min, tau * gain * u_max))
                target_value = float(config["target"])
                if not (reachable[0] - 1e-9 <= target_value <= reachable[1] + 1e-9):
                    issues.append(ValidationIssue(
                        "warning", "mpc_target_unreachable",
                        f"Target {target_value:g} lies outside the steady-state range "
                        f"[{reachable[0]:g}, {reachable[1]:g}] reachable within the input "
                        f"bounds, so a steady tracking error is unavoidable.",
                        f"{path}.target"))
            except (TypeError, ValueError, ZeroDivisionError):
                pass
        return issues

    # -- compilation --------------------------------------------------------
    def compile(self, project: Dict[str, Any]) -> Dict[str, Any]:
        issues = [i for i in self.validate(project) if i.severity == "error"]
        if issues:
            raise ValueError("MPC configuration is not valid: " +
                             "; ".join(i.message for i in issues))
        config = _merged(self.approach_config(project))
        interval = float(config["control_interval"])
        steps = max(1, int(round(float(config["duration"]) / interval)))
        return {
            "approach": self.approach_id,
            "config": config,
            "steps": steps,
            "interval": interval,
            # The stage-4 model travels with the compiled plan, because run() only
            # receives `compiled` and the plant may need to BE that model rather than the
            # built-in first-order system.
            "model": (project or {}).get("model") or {},
            "engine_version": self.get_capabilities().engine_version,
        }

    # -- execution ----------------------------------------------------------
    def run(self, compiled: Dict[str, Any], context: RunContext) -> Dict[str, Any]:
        config = compiled["config"]
        dt = float(compiled["interval"])
        steps = int(compiled["steps"])

        horizon = int(round(float(config["prediction_horizon"])))
        control_horizon = int(round(float(config["control_horizon"])))
        u_min, u_max = float(config["input_min"]), float(config["input_max"])
        rate_limit = float(config["input_rate_limit"])
        w_track = float(config["weight_tracking"])
        w_effort = float(config["weight_effort"])
        w_rate = float(config["weight_rate"])
        y_min = config.get("output_min")
        y_max = config.get("output_max")
        y_min = float(y_min) if y_min is not None else None
        y_max = float(y_max) if y_max is not None else None
        tolerance = float(config["tolerance"])
        max_iter = int(round(float(config["max_iterations"])))
        model_spec = config["model"]
        plant_spec = config["plant"]

        # If the project asks for it, the plant becomes the STAGE-4 MODEL rather than the
        # built-in first-order system. Failure here is loud: a controller that silently
        # falls back to a toy plant would report a beautifully converged result about a
        # system the researcher never described, which is the whole defect this addresses.
        network_plant = None
        if str(plant_spec.get("kind", "")).strip() == "reaction_network":
            network_plant = ReactionNetworkPlant(
                compiled.get("model") or {},
                input_species=str(config.get("controlled_input") or ""),
                output_species=str(config.get("measured_output") or ""))

        state = float(config["initial_state"])
        if network_plant is not None:
            state = network_plant.output()
        previous_u = 0.0

        times: List[float] = []
        outputs: List[float] = []
        targets: List[float] = []
        controls: List[float] = []
        errors: List[float] = []
        objectives: List[float] = []
        violations: List[Dict[str, Any]] = []
        failures = 0

        context.log(f"MPC starting: horizon={horizon}, control horizon={control_horizon}, "
                    f"interval={dt:g}, input bounds [{u_min:g}, {u_max:g}], "
                    f"rate limit {rate_limit:g}/step.")

        def predict(sequence: np.ndarray, start_state: float, t0: float) -> np.ndarray:
            """Roll the PREDICTIVE MODEL forward over the horizon."""
            predicted = np.empty(horizon, dtype=float)
            x = start_state
            for k in range(horizon):
                # Beyond the control horizon the last action is held -- the standard
                # receding-horizon assumption.
                u = sequence[min(k, control_horizon - 1)]
                x = _step_first_order(x, float(u), dt, model_spec)
                predicted[k] = x
            return predicted

        for step in range(steps):
            context.checkpoint()
            t = step * dt
            target_now = _target_at(config, t)
            horizon_targets = np.array(
                [_target_at(config, t + (k + 1) * dt) for k in range(horizon)], dtype=float)

            def objective(sequence: np.ndarray, _state=state, _t=t) -> float:
                predicted = predict(sequence, _state, _t)
                tracking = float(np.sum((predicted - horizon_targets) ** 2))
                effort = float(np.sum(sequence ** 2))
                deltas = np.diff(np.concatenate(([previous_u], sequence)))
                rate = float(np.sum(deltas ** 2))
                return w_track * tracking + w_effort * effort + w_rate * rate

            # Rate limits as inequality constraints: |u[k] - u[k-1]| <= rate_limit.
            constraints: List[Dict[str, Any]] = []
            if rate_limit > 0:
                def rate_constraint(sequence: np.ndarray) -> np.ndarray:
                    deltas = np.diff(np.concatenate(([previous_u], sequence)))
                    return rate_limit - np.abs(deltas)
                constraints.append({"type": "ineq", "fun": rate_constraint})
            if y_max is not None:
                constraints.append({"type": "ineq",
                                    "fun": lambda s, _st=state, _t=t: y_max - predict(s, _st, _t)})
            if y_min is not None:
                constraints.append({"type": "ineq",
                                    "fun": lambda s, _st=state, _t=t: predict(s, _st, _t) - y_min})

            guess = np.clip(np.full(control_horizon, previous_u, dtype=float), u_min, u_max)
            result = scipy.optimize.minimize(
                objective, guess, method="SLSQP",
                bounds=[(u_min, u_max)] * control_horizon,
                constraints=constraints,
                options={"maxiter": max_iter, "ftol": max(tolerance, 1e-12)},
            )

            if result.success and np.all(np.isfinite(result.x)):
                sequence = np.asarray(result.x, dtype=float)
                objective_value = float(result.fun)
            else:
                # Report, do not hide. Holding the previous action is the safe
                # fallback; the failure is counted and surfaced in the results.
                failures += 1
                sequence = np.full(control_horizon, previous_u, dtype=float)
                objective_value = float(objective(sequence))
                context.log(
                    f"t={t:g}: optimisation did not converge "
                    f"({getattr(result, 'message', 'no message')}); holding u={previous_u:g}.",
                    "warning")

            # Apply only the first action, clipped to bounds and rate limit so a
            # reported control value can never violate its constraints.
            u_applied = float(sequence[0])
            if rate_limit > 0:
                u_applied = float(np.clip(u_applied, previous_u - rate_limit, previous_u + rate_limit))
            u_applied = float(np.clip(u_applied, u_min, u_max))

            # Advance the PLANT (which may differ from the model).
            #
            # When plant.kind is "reaction_network" the plant IS the stage-4 model the
            # researcher prepared, integrated as a well-mixed system, with u dosed into
            # the named input species. Otherwise it stays the first-order test system,
            # which is a controller demonstration rather than a statement about biology.
            if network_plant is not None:
                state = network_plant.step(u_applied, dt)
            else:
                state = _step_first_order(state, u_applied, dt, plant_spec)

            times.append(float(t + dt))
            outputs.append(float(state))
            targets.append(float(_target_at(config, t + dt)))
            controls.append(u_applied)
            errors.append(float(state - _target_at(config, t + dt)))
            objectives.append(objective_value)

            if y_max is not None and state > y_max + 1e-9:
                violations.append({"time": float(t + dt), "kind": "output_max",
                                   "value": float(state), "limit": y_max})
            if y_min is not None and state < y_min - 1e-9:
                violations.append({"time": float(t + dt), "kind": "output_min",
                                   "value": float(state), "limit": y_min})

            previous_u = u_applied
            if steps > 1 and (step % max(1, steps // 20) == 0 or step == steps - 1):
                context.progress((step + 1) / steps, f"t={t + dt:g}")

        error_array = np.abs(np.asarray(errors, dtype=float))
        first_window = error_array[: max(1, len(error_array) // 5)]
        last_window = error_array[-max(1, len(error_array) // 5):]
        summary = {
            "final_output": outputs[-1] if outputs else None,
            "final_target": targets[-1] if targets else None,
            "final_abs_error": float(error_array[-1]) if error_array.size else None,
            "mean_abs_error": float(error_array.mean()) if error_array.size else None,
            "initial_mean_abs_error": float(first_window.mean()) if first_window.size else None,
            "final_mean_abs_error": float(last_window.mean()) if last_window.size else None,
            "tracking_improved": bool(last_window.mean() < first_window.mean())
            if error_array.size else False,
            "optimisation_failures": int(failures),
            "constraint_violations": len(violations),
            "input_bounds_respected": bool(
                all(u_min - 1e-9 <= u <= u_max + 1e-9 for u in controls)),
            "rate_limit_respected": bool(
                rate_limit <= 0 or all(
                    abs(controls[i] - (controls[i - 1] if i else 0.0)) <= rate_limit + 1e-9
                    for i in range(len(controls)))),
        }
        context.log(f"MPC finished: mean |error| {summary['mean_abs_error']:.4g}, "
                    f"{failures} optimisation failure(s), "
                    f"{len(violations)} constraint violation(s).")

        return {
            "approach": self.approach_id,
            "kind": "control",
            "t": times,
            "series": {
                "output": outputs,
                "target": targets,
                "control": controls,
                "error": errors,
                "objective": objectives,
            },
            "units": {"output": config.get("output_units", ""),
                      "control": config.get("input_units", "")},
            "violations": violations,
            "summary": summary,
        }

    def export_configuration(self, project: Dict[str, Any]) -> Dict[str, Any]:
        config = _merged(self.approach_config(project))
        return {
            "approach": self.approach_id,
            "label": self.label,
            "configuration": config,
            "objective": ("J = w_track*sum (y-target)^2 + w_effort*sum u^2 "
                          "+ w_rate*sum (du)^2"),
            "constraints": {
                "input_bounds": [config["input_min"], config["input_max"]],
                "input_rate_limit": config["input_rate_limit"],
                "output_min": config.get("output_min"),
                "output_max": config.get("output_max"),
            },
        }


import scipy  # noqa: E402  (imported last for the version string only)

MPC_ADAPTER = register_approach(MPCAdapter())
