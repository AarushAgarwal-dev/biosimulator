#!/usr/bin/env python3
"""Numerical verification of the BioSimulator ODE / PDE engines.

Runs eight reference models through the project's OWN engine modules and checks each
against a property the mathematics guarantees: a closed-form solution, a conservation
law, an analytic steady state, a sign condition, or a growth condition. Absence of an
exception is never treated as success -- every model reports a measured number, the
tolerance it was compared against, and PASS or FAIL.

Engines exercised
    simulation_engine.ODEModel   symbolic ODE compiler + LSODA integration
    simulation_engine.solve_pde  2D explicit reaction-diffusion
    pde_solver_1d.solve_1d       1D reaction-diffusion-advection with real boundaries
    pde_model                    field presets, safe expression parsing, compilation
    boundary_conditions          named Dirichlet / Neumann conditions
    geometry + meshing           interval domain -> tagged 1D mesh

Run:  python verify_models.py      Exit code 0 only if every check passes.
No network, no LLM, no pytest.
"""

from __future__ import annotations

import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import scipy.optimize
import sympy as sp

import boundary_conditions as bc
import geometry as geo
import meshing
import pde_model
from pde_solver_1d import solve_1d, steady_state_1d
from simulation_engine import ODEModel, solve_pde

# Deterministic: solve_pde seeds its noisy initial conditions from the global RNG.
SEED = 20260913

# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------
RESULTS: List[Dict[str, Any]] = []

_OPS: Dict[str, Callable[[float, float], bool]] = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
}


def check(model: str, what: str, measured: float, op: str, limit: float) -> bool:
    """Compare one measured quantity against one stated tolerance and record it."""
    value = float(measured)
    passed = bool(np.isfinite(value) and _OPS[op](value, limit))
    RESULTS.append({"model": model, "what": what, "measured": f"{value:.4e}",
                    "tolerance": f"{op} {limit:.1e}", "passed": passed})
    print(f"      {'PASS' if passed else 'FAIL'}  {what:<52s} "
          f"{value:>12.4e}   (tolerance {op} {limit:.1e})")
    return passed


def note(text: str) -> None:
    print(f"      ....  {text}")


def record_crash(model: str, exc: BaseException) -> None:
    message = f"model raised {type(exc).__name__}: {exc}"
    RESULTS.append({"model": model, "what": message[:120], "measured": "n/a",
                    "tolerance": "no exception", "passed": False})
    print(f"      FAIL  {message}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def ode_blueprint(nodes: Dict[str, float], parameters: Dict[str, float],
                  odes: Dict[str, str], fluxes: Optional[Dict[str, str]] = None,
                  t_max: float = 10.0) -> Dict[str, Any]:
    """A custom-kinetics blueprint: explicit rate laws, optional shared fluxes."""
    blueprint: Dict[str, Any] = {
        "type": "ODE",
        "nodes": [{"id": nid, "name": nid, "initial_value": float(v)}
                  for nid, v in nodes.items()],
        "parameters": dict(parameters),
        "odes": dict(odes),
        "simulation_config": {"t_max": float(t_max)},
    }
    if fluxes:
        blueprint["fluxes"] = dict(fluxes)
    return blueprint


def stable_dt(dx: float, diffusion: float, velocity: float = 0.0,
              safety: float = 0.4) -> float:
    """Forward-Euler step inside BOTH the diffusive and advective limits.

    safety <= 0.5 also keeps the update a convex combination of neighbours, which is
    what makes the discrete maximum principle (checks in model 6) hold.
    """
    limits = []
    if diffusion > 0:
        limits.append(dx * dx / (2.0 * diffusion))
    if abs(velocity) > 0:
        limits.append(dx / abs(velocity))
    return safety * min(limits) if limits else 1e-3


def build_1d(preset: str, length: float, elements: int,
             field_overrides: Optional[Dict[str, Any]] = None,
             parameter_overrides: Optional[Dict[str, float]] = None):
    """geometry -> meshing -> pde_model: the project's real 1D setup path."""
    model = pde_model.get_preset(preset)
    if parameter_overrides:
        model["parameters"].update(parameter_overrides)
    if field_overrides:
        model["fields"][0] = dict(model["fields"][0], **field_overrides)
    name = str(model["fields"][0]["name"])

    domain = geo.make_interval(length, units="mm", name="Verification interval")
    problems = [i for i in geo.validate_domain(domain) if i.severity == "error"]
    if problems:
        raise RuntimeError("domain invalid: " + "; ".join(i.message for i in problems))

    mesh = meshing.mesh_1d(domain, element_count=elements)
    problems = [i for i in meshing.validate_mesh(mesh, domain) if i.severity == "error"]
    if problems:
        raise RuntimeError("mesh invalid: " + "; ".join(i.message for i in problems))

    x = np.array([node[0] for node in mesh["nodes"]], dtype=float)
    compiled = pde_model.compile_field(model, name, domain)
    zeros = np.zeros_like(x)
    u0 = np.asarray(compiled["initial"](x, zeros), dtype=float)
    return model, domain, mesh, x, compiled, u0


def reaction_adapter(compiled: Dict[str, Any]) -> Callable:
    """compile_field gives reaction(fields..., x, y, t); solve_1d wants (u, x, t)."""
    def reaction(u, xs, t):
        return np.asarray(compiled["reaction"](u, xs, np.zeros_like(xs), t), dtype=float)
    return reaction


def source_adapter(compiled: Dict[str, Any]) -> Callable:
    def source(xs, t):
        return np.asarray(compiled["source"](xs, np.zeros_like(xs), t), dtype=float)
    return source


def centroid(x: np.ndarray, u: np.ndarray) -> float:
    total = float(np.sum(u))
    return float(np.sum(x * u) / total) if abs(total) > 1e-300 else float("nan")


def dominant_mode(jac: np.ndarray, diffusion: np.ndarray, nx: int, ny: int,
                  dx: float, dy: float) -> Dict[str, Any]:
    """Fastest-growing mode of the DISCRETISED reaction-diffusion operator.

    The 5-point Laplacian with zero-flux edges has eigenvalues
    ``-4 sin^2(m*pi/(2N)) / dx^2``, so mode (m, n) grows at the largest real
    eigenvalue of ``J + diag(D) * (lambda_x + lambda_y)``. Working from the DISCRETE
    spectrum rather than the continuum ``k^2`` makes the prediction an exact statement
    about what the solver actually integrates, with no truncation error of its own.
    """
    lam_x = -4.0 * np.sin(np.arange(nx) * np.pi / (2 * nx)) ** 2 / (dx * dx)
    lam_y = -4.0 * np.sin(np.arange(ny) * np.pi / (2 * ny)) ** 2 / (dy * dy)
    laplacian = (lam_x[:, None] + lam_y[None, :]).reshape(-1, 1, 1)
    operators = jac[None, :, :] + np.diag(diffusion)[None, :, :] * laplacian
    growth = np.max(np.real(np.linalg.eigvals(operators)), axis=1).reshape(nx, ny)
    mi, ni = (int(v) for v in np.unravel_index(int(np.argmax(growth)), growth.shape))
    wavenumber = float(np.hypot(mi * np.pi / (nx * dx), ni * np.pi / (ny * dy)))
    return {"growth_rate": float(growth[mi, ni]), "mode": (mi, ni),
            "wavelength": 2.0 * np.pi / max(wavenumber, 1e-30),
            "cells_per_wavelength": (2.0 * np.pi / max(wavenumber, 1e-30)) / dx}


# ---------------------------------------------------------------------------
# 1. ODE exponential decay, against the closed form
# ---------------------------------------------------------------------------
def model_exponential_decay() -> None:
    name = "ODE exponential decay"
    u0, k, t_max = 2.0, 0.7, 3.0
    print("      du/dt = -k*u     exact: u(t) = u0*exp(-k*t)")
    print("      engine: simulation_engine.ODEModel -> LSODA (rtol 1e-8, atol 1e-10)")
    print(f"      setup : u0={u0}, k={k}, t in [0, {t_max}], 61 sample times")

    result = ODEModel(ode_blueprint({"u": u0}, {"k": k}, {"u": "-k*u"},
                                    t_max=t_max)).simulate(t_max=t_max, num_points=61)
    t = np.asarray(result["t"], dtype=float)
    u = np.asarray(result["species"]["u"], dtype=float)
    exact = u0 * np.exp(-k * t)

    check(name, "max |u_numeric - u0*exp(-k t)|", np.max(np.abs(u - exact)), "<", 1e-6)
    check(name, "max relative error", np.max(np.abs(u - exact) / exact), "<", 1e-6)
    check(name, "non-finite sample count", np.sum(~np.isfinite(u)), "<=", 0)


# ---------------------------------------------------------------------------
# 2. ODE logistic growth: approaches K, never exceeds it
# ---------------------------------------------------------------------------
def model_logistic_growth() -> None:
    name = "ODE logistic growth"
    u0, r, K, t_max = 0.1, 1.0, 5.0, 20.0
    print("      du/dt = r*u*(1 - u/K)     exact: u = K*u0*e^{rt} / (K + u0*(e^{rt}-1))")
    print(f"      setup : u0={u0}, r={r}, K={K}, t in [0, {t_max}], 201 sample times")

    result = ODEModel(ode_blueprint({"u": u0}, {"r": r, "K": K},
                                    {"u": "r*u*(1 - u/K)"}, t_max=t_max)
                      ).simulate(t_max=t_max, num_points=201)
    t = np.asarray(result["t"], dtype=float)
    u = np.asarray(result["species"]["u"], dtype=float)
    growth = np.exp(r * t)
    exact = K * u0 * growth / (K + u0 * (growth - 1.0))

    note(f"final value {u[-1]:.10f} against carrying capacity K={K}")
    check(name, "|u(t_end) - K| (approach to capacity)", abs(u[-1] - K), "<", 1e-4)
    # A logistic trajectory started below K can never reach it, let alone pass it; the
    # only slack allowed is integrator round-off, not a physically impossible overshoot.
    check(name, "max(u) - K (negative = never exceeded)", np.max(u) - K, "<=", 1e-9)
    check(name, "min du between samples (monotone increase)",
          float(np.min(np.diff(u))), ">=", -1e-12)
    check(name, "max |u_numeric - closed form|", np.max(np.abs(u - exact)), "<", 1e-6)


# ---------------------------------------------------------------------------
# 3. Two-species Michaelis-Menten turnover through a SHARED flux
# ---------------------------------------------------------------------------
def model_michaelis_menten() -> None:
    name = "ODE Michaelis-Menten"
    s0, p0, vmax, km, t_max = 10.0, 0.0, 1.0, 2.0, 8.0
    print("      v = Vmax*S/(Km + S);   dS/dt = -v,  dP/dt = +v   (shared flux)")
    print(f"      setup : S0={s0}, P0={p0}, Vmax={vmax}, Km={km}, t in [0, {t_max}]")

    result = ODEModel(ode_blueprint({"S": s0, "P": p0}, {"Vmax": vmax, "Km": km},
                                    {"S": "-v", "P": "v"},
                                    fluxes={"v": "Vmax*S/(Km + S)"}, t_max=t_max)
                      ).simulate(t_max=t_max, num_points=161)
    S = np.asarray(result["species"]["S"], dtype=float)
    P = np.asarray(result["species"]["P"], dtype=float)

    note(f"S: {s0:.4f} -> {S[-1]:.6f},  P: {p0:.4f} -> {P[-1]:.6f}")
    check(name, "non-finite sample count (both species)",
          np.sum(~np.isfinite(S)) + np.sum(~np.isfinite(P)), "<=", 0)
    check(name, "min concentration over both species",
          min(float(S.min()), float(P.min())), ">=", -1e-9)
    check(name, "max |(S+P) - (S0+P0)| (shared flux conserves)",
          np.max(np.abs((S + P) - (s0 + p0))), "<", 1e-6)


# ---------------------------------------------------------------------------
# 4. 1D diffusion, Dirichlet ends at different values -> straight line
# ---------------------------------------------------------------------------
def model_dirichlet_steady_state() -> None:
    name = "1D diffusion, Dirichlet"
    left_value, right_value, length, elements, t_max = 2.0, 5.0, 1.0, 40, 3.0
    print("      du/dt = D d2u/dx2, u(0)=2, u(L)=5    exact steady state: a straight line")
    print("      engine: pde_model preset 'diffusion' -> boundary_conditions -> solve_1d")

    _, domain, _, x, compiled, u0 = build_1d(
        "diffusion", length, elements,
        field_overrides={"initial": "0.0", "units": "mM",
                         "t_end": t_max, "output_interval": 0.25})
    conditions = [bc.make_dirichlet("u", "left", left_value),
                  bc.make_dirichlet("u", "right", right_value)]
    issues = bc.validate_conditions(conditions, domain, ["u"])
    check(name, "boundary-condition validation errors",
          sum(1 for i in issues if i.severity == "error"), "<=", 0)
    resolved = bc.resolve_1d_conditions(conditions, "u")

    dx = float(x[1] - x[0])
    dt = stable_dt(dx, compiled["diffusion"])
    note(f"{elements} elements, dx={dx:g}, dt={dt:.3e}, D={compiled['diffusion']:g}, "
         f"t_max={t_max}")
    result = solve_1d(x, u0, diffusion=compiled["diffusion"], t_max=t_max, dt=dt,
                      reaction=reaction_adapter(compiled),
                      source=source_adapter(compiled),
                      left=resolved["left"], right=resolved["right"],
                      save_every=2000)
    final = np.asarray(result["u"][-1], dtype=float)
    analytic = steady_state_1d(x, compiled["diffusion"], left_value, right_value)

    check(name, "std of first differences (0 for a straight line)",
          np.std(np.diff(final)), "<", 1e-6)
    check(name, "max |u_final - analytic straight line|",
          np.max(np.abs(final - analytic)), "<", 1e-6)
    check(name, "|u[0] - 2| + |u[-1] - 5|  (ends held exactly)",
          abs(final[0] - left_value) + abs(final[-1] - right_value), "<", 1e-12)


# ---------------------------------------------------------------------------
# 5. 1D diffusion, no-flux (Neumann) ends -> mass conserved
# ---------------------------------------------------------------------------
def model_noflux_mass_conservation() -> None:
    name = "1D diffusion, no-flux"
    length, elements, t_max = 1.0, 40, 0.2
    print("      du/dt = D d2u/dx2 with -D du/dn = 0 at both ends: mass is invariant")
    print("      engine: pde_model preset 'diffusion' (Gaussian initial) -> solve_1d")

    _, domain, _, x, compiled, u0 = build_1d(
        "diffusion", length, elements,
        field_overrides={"units": "mM", "t_end": t_max, "output_interval": 0.02})
    conditions = [bc.make_no_flux("u", "left"), bc.make_no_flux("u", "right")]
    issues = bc.validate_conditions(conditions, domain, ["u"])
    check(name, "boundary-condition validation errors",
          sum(1 for i in issues if i.severity == "error"), "<=", 0)
    resolved = bc.resolve_1d_conditions(conditions, "u")

    dx = float(x[1] - x[0])
    dt = stable_dt(dx, compiled["diffusion"])
    result = solve_1d(x, u0, diffusion=compiled["diffusion"], t_max=t_max, dt=dt,
                      reaction=reaction_adapter(compiled),
                      source=source_adapter(compiled),
                      left=resolved["left"], right=resolved["right"], save_every=100)
    mass = np.asarray(result["mass"], dtype=float)
    frames = np.asarray(result["u"], dtype=float)
    note(f"{len(mass)} saved frames, mass {mass[0]:.12f} -> {mass[-1]:.12f} "
         f"(trapezoidal integral)")

    check(name, "max |mass(t)/mass(0) - 1| over all frames",
          np.max(np.abs(mass / mass[0] - 1.0)), "<", 1e-10)
    check(name, "peak decay (diffusion must flatten the profile)",
          float(frames[0].max() - frames[-1].max()), ">", 0.0)
    check(name, "non-finite values in any frame", np.sum(~np.isfinite(frames)), "<=", 0)


# ---------------------------------------------------------------------------
# 6. 1D advection-diffusion: centroid follows the flow, profile stays bounded
# ---------------------------------------------------------------------------
def model_advection_diffusion() -> None:
    name = "1D advection-diffusion"
    length, elements, t_max = 2.0, 80, 0.5
    print("      du/dt = D d2u/dx2 - v du/dx    upwind advection, no-flux ends")
    print("      engine: pde_model preset 'advection_diffusion' -> solve_1d")

    for velocity, start in ((1.0, 0.3), (-1.0, 1.7)):
        label = f"v={velocity:+g}"
        _, domain, _, x, compiled, u0 = build_1d(
            "advection_diffusion", length, elements,
            field_overrides={"units": "mM", "t_end": t_max, "output_interval": 0.05,
                             "initial": f"exp(-((x-{start})**2)/0.005)"},
            parameter_overrides={"v": velocity})
        resolved = bc.resolve_1d_conditions(
            [bc.make_no_flux("u", "left"), bc.make_no_flux("u", "right")], "u")

        dx = float(x[1] - x[0])
        vx = compiled["advection"]["vx"]
        dt = stable_dt(dx, compiled["diffusion"], vx)
        result = solve_1d(x, u0, diffusion=compiled["diffusion"], t_max=t_max, dt=dt,
                          advection=vx, reaction=reaction_adapter(compiled),
                          source=source_adapter(compiled),
                          left=resolved["left"], right=resolved["right"], save_every=5)
        frames = np.asarray(result["u"], dtype=float)
        moved = centroid(x, frames[-1]) - centroid(x, frames[0])
        expected = vx * t_max
        note(f"{label}: centroid {centroid(x, frames[0]):.5f} -> "
             f"{centroid(x, frames[-1]):.5f} (moved {moved:+.5f}, expected "
             f"{expected:+.5f}); numerical diffusion "
             f"{result['stability']['numerical_diffusion']:.4g}")

        # Direction: displacement must share the sign of the velocity.
        check(name, f"{label}: signed displacement / velocity", moved / vx, ">", 0.0)
        check(name, f"{label}: |displacement - v*t|", abs(moved - expected), "<", 2e-2)
        # Upwind differencing is monotone: no negative undershoot and no new maximum.
        check(name, f"{label}: min value in any frame", float(frames.min()), ">=", -1e-12)
        check(name, f"{label}: max(u) - max(u at t=0)",
              float(frames.max() - frames[0].max()), "<=", 1e-12)


# ---------------------------------------------------------------------------
# 7. 2D reaction-diffusion: the repo's own Turing preset
# ---------------------------------------------------------------------------
def model_turing_pattern() -> None:
    name = "2D Turing pattern"
    import agent  # rule_based_parse holds the repo's default Turing blueprint

    blueprint = agent.rule_based_parse(
        "A reaction-diffusion Turing system forming a spatial pattern.")
    if blueprint.get("type") != "PDE":
        raise RuntimeError("the repo's Turing preset did not come back as a PDE blueprint")
    spatial = blueprint["spatial"]
    reactions = spatial["reactions"]
    t_max = float(blueprint["simulation_config"]["t_max"])
    dt = float(blueprint["simulation_config"]["dt"])
    print("      Gierer-Meinhardt activator-inhibitor, 2D 5-point Laplacian, zero flux")
    print("      engine: agent.rule_based_parse Turing preset -> simulation_engine.solve_pde")
    print(f"      setup : {spatial['x_grid']}x{spatial['y_grid']} grid, D={spatial['diffusion']}, "
          f"t_max={t_max}, dt={dt}")
    for species, formula in reactions.items():
        print(f"              d{species}/dt = D*lap({species}) + {formula}")

    # Homogeneous steady state of the preset's OWN reaction terms, found numerically,
    # so the run starts from equilibrium and any pattern is the instability itself.
    symbols = {n: sp.Symbol(n) for n in reactions}
    order = list(reactions)
    exprs = [sp.sympify(reactions[n], locals=symbols) for n in order]
    rhs = sp.lambdify([symbols[n] for n in order], exprs, "numpy")
    root = scipy.optimize.fsolve(lambda z: np.asarray(rhs(*z), dtype=float),
                                 np.ones(len(order)), full_output=False)
    residual = float(np.max(np.abs(np.asarray(rhs(*root), dtype=float))))
    steady = {n: float(v) for n, v in zip(order, root)}
    note("homogeneous steady state " +
         ", ".join(f"{n}={v:.6f}" for n, v in steady.items()))
    check(name, "residual |R(u*)| at the homogeneous state", residual, "<", 1e-9)

    # ---- Linear stability of the DISCRETISED problem -----------------------
    jacobian = sp.Matrix([[sp.diff(expr, symbols[n]) for n in order] for expr in exprs])
    J = np.array(jacobian.subs({symbols[n]: steady[n] for n in order}).evalf(),
                 dtype=float)
    nx, ny = int(spatial["x_grid"]), int(spatial["y_grid"])
    dx, dy = float(spatial["dx"]), float(spatial["dy"])
    diffusivity = np.array([float(spatial["diffusion"][n]) for n in order])
    shipped = dominant_mode(J, diffusivity, nx, ny, dx, dy)
    sigma_max = shipped["growth_rate"]
    note("reaction Jacobian at u*: " +
         "; ".join("[" + ", ".join(f"{v:+.6f}" for v in row) + "]" for row in J))
    note(f"fastest-growing discrete mode (m,n)={shipped['mode']}, growth rate "
         f"{sigma_max:+.5f} /time, wavelength {shipped['wavelength']:.4f} = "
         f"{shipped['cells_per_wavelength']:.2f} cells")
    check(name, "predicted growth rate of the fastest mode (unstable)",
          sigma_max, ">", 0.0)

    # Is that wavelength set by the biology or by the mesh? Refine the grid at the
    # SAME physical extent and see whether the selected wavelength stops moving.
    # Linear algebra only -- no extra PDE solves.
    extent_x, extent_y = nx * dx, ny * dy
    refined = []
    for factor in (2, 4, 8):
        fine = dominant_mode(J, diffusivity, nx * factor, ny * factor,
                             dx / factor, dy / factor)
        refined.append(fine)
        note(f"refinement x{factor}: dx={dx / factor:g}, wavelength "
             f"{fine['wavelength']:.4f} ({fine['cells_per_wavelength']:.2f} cells), "
             f"growth rate {fine['growth_rate']:+.5f}")
    converged = refined[-1]["wavelength"]
    check(name, "wavelength convergence |L(x4)/L(x8) - 1|",
          abs(refined[-2]["wavelength"] / converged - 1.0), "<", 0.05)
    shipped_error = abs(shipped["wavelength"] / converged - 1.0)
    note(f"WARNING: the shipped preset's grid (dx={dx:g}, extent {extent_x:g}x{extent_y:g})"
         f" selects a wavelength {shipped_error * 100:.1f}% away from the mesh-converged"
         f" value {converged:.4f}; at {shipped['cells_per_wavelength']:.2f} cells per"
         f" wavelength the pattern SCALE is set by the mesh, not by the kinetics."
         f" A preset-resolution issue, not a solver error.")

    # ---- Linear phase: measured growth rate against that prediction --------
    np.random.seed(SEED)
    initial = {n: {"type": "random_noise", "base_value": steady[n],
                   "noise_amplitude": 0.02} for n in order}
    linear_end = min(t_max, 12.0)
    linear = solve_pde(spatial, reactions, initial, t_max=linear_end, dt=dt, save_every=5)
    lin_frames = np.asarray(linear["species"][order[0]], dtype=float)
    lin_times = np.asarray(linear["t"], dtype=float)
    lin_amp = np.array([float(frame.std()) for frame in lin_frames])
    # Fit only where the unstable mode already dominates but nothing has saturated:
    # a 5% perturbation of u* is still firmly in the linear regime.
    window = (lin_amp > 2.0 * lin_amp[0]) & (lin_amp < 0.05)
    if int(window.sum()) < 3:
        raise RuntimeError(f"only {int(window.sum())} frames fell in the linear window; "
                           f"amplitudes {lin_amp[:8]}")
    measured_rate = float(np.polyfit(lin_times[window], np.log(lin_amp[window]), 1)[0])
    note(f"linear phase fitted over {int(window.sum())} frames in "
         f"t=[{lin_times[window][0]:g}, {lin_times[window][-1]:g}]: measured growth rate "
         f"{measured_rate:+.5f} /time against predicted {sigma_max:+.5f}")
    check(name, "|measured/predicted growth rate - 1|",
          abs(measured_rate / sigma_max - 1.0), "<", 0.2)

    # ---- Full preset run: finiteness and pattern formation -----------------
    np.random.seed(SEED)
    result = solve_pde(spatial, reactions, initial, t_max=t_max, dt=dt, save_every=50)
    frames = {n: np.asarray(result["species"][n], dtype=float) for n in order}
    non_finite = sum(int(np.sum(~np.isfinite(frames[n]))) for n in order)
    amplitude = np.array([float(frame.std()) for frame in frames[order[0]]])
    growth = amplitude[-1] / amplitude[0]
    stability = result["stability"]
    note(f"{len(result['t'])} saved frames to t={result['t'][-1]:g}; diffusion number "
         f"{stability['diffusion_number']:.3f} (limit 0.5); diverged_at "
         f"{stability['diverged_at']}")
    note(f"{order[0]} spatial std {amplitude[0]:.6f} -> {amplitude[-1]:.6f}; final range "
         f"[{frames[order[0]][-1].min():.4f}, {frames[order[0]][-1].max():.4f}]")

    check(name, "non-finite values across all frames/species", non_finite, "<=", 0)
    check(name, "pattern amplitude growth (std_final / std_0)", growth, ">=", 10.0)
    check(name, "amplitude at the final frame", amplitude[-1], ">", 0.05)
    check(name, "reported divergence flag (0 = none)",
          0 if stability["diverged_at"] is None else 1, "<=", 0)


# ---------------------------------------------------------------------------
# 8. Lotka-Volterra: positivity and the conserved invariant
# ---------------------------------------------------------------------------
def model_lotka_volterra() -> None:
    name = "ODE Lotka-Volterra"
    a, b, g, d = 1.1, 0.4, 0.4, 0.1
    x0, y0, t_max = 10.0, 5.0, 20.0
    print("      dX/dt = a*X - b*X*Y,  dY/dt = d*X*Y - g*Y")
    print("      invariant: V = d*X - g*ln(X) + b*Y - a*ln(Y) is constant in exact time")
    print(f"      setup : X0={x0}, Y0={y0}, a={a}, b={b}, g={g}, d={d}, t in [0, {t_max}]")

    result = ODEModel(ode_blueprint({"X": x0, "Y": y0},
                                    {"a": a, "b": b, "g": g, "d": d},
                                    {"X": "a*X - b*X*Y", "Y": "d*X*Y - g*Y"},
                                    t_max=t_max)).simulate(t_max=t_max, num_points=401)
    X = np.asarray(result["species"]["X"], dtype=float)
    Y = np.asarray(result["species"]["Y"], dtype=float)

    check(name, "non-finite sample count (both species)",
          np.sum(~np.isfinite(X)) + np.sum(~np.isfinite(Y)), "<=", 0)
    check(name, "smallest population reached (must stay positive)",
          min(float(X.min()), float(Y.min())), ">", 0.0)
    invariant = d * X - g * np.log(X) + b * Y - a * np.log(Y)
    drift = float(np.max(np.abs(invariant - invariant[0])) / abs(invariant[0]))
    note(f"X in [{X.min():.4f}, {X.max():.4f}], Y in [{Y.min():.4f}, {Y.max():.4f}]; "
         f"invariant {invariant[0]:.8f} +/- {np.max(np.abs(invariant - invariant[0])):.2e}")
    check(name, "max relative drift of the conserved invariant", drift, "<", 1e-4)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
MODELS: List[tuple] = [
    ("ODE exponential decay", model_exponential_decay),
    ("ODE logistic growth", model_logistic_growth),
    ("ODE Michaelis-Menten", model_michaelis_menten),
    ("1D diffusion, Dirichlet ends", model_dirichlet_steady_state),
    ("1D diffusion, no-flux ends", model_noflux_mass_conservation),
    ("1D advection-diffusion", model_advection_diffusion),
    ("2D Turing reaction-diffusion", model_turing_pattern),
    ("ODE Lotka-Volterra", model_lotka_volterra),
]

RULE = "=" * 112
THIN = "-" * 112


def main() -> int:
    started = time.time()
    print(RULE)
    print(" BioSimulator engine verification -- ODE and PDE numerics")
    print(f" python {sys.version.split()[0]}   numpy {np.__version__}   "
          f"scipy {scipy.__version__}   sympy {sp.__version__}")
    print(" Each model is checked against a mathematical guarantee, not merely for the")
    print(" absence of an exception. Exit code 0 only if every check passes.")
    print(RULE)

    for index, (title, function) in enumerate(MODELS, start=1):
        print(f"\n[{index}/{len(MODELS)}] {title}")
        try:
            function()
        except Exception as exc:                      # a crash is a failure, not a skip
            record_crash(title, exc)

    passed = sum(1 for r in RESULTS if r["passed"])
    failed = len(RESULTS) - passed

    print(f"\n{RULE}")
    print(" SUMMARY")
    print(THIN)
    print(f" {'MODEL':<24s} {'CHECK':<52s} {'MEASURED':>12s} {'TOLERANCE':>12s}  VERDICT")
    print(THIN)
    for row in RESULTS:
        print(f" {row['model'][:24]:<24s} {row['what'][:52]:<52s} "
              f"{row['measured']:>12s} {row['tolerance']:>12s}  "
              f"{'PASS' if row['passed'] else 'FAIL'}")
    print(THIN)
    print(f" {len(RESULTS)} checks over {len(MODELS)} models: {passed} passed, "
          f"{failed} failed, in {time.time() - started:.1f}s")
    if failed:
        print(" FAILING CHECKS:")
        for row in RESULTS:
            if not row["passed"]:
                print(f"   - {row['model']}: {row['what']} "
                      f"(measured {row['measured']}, needed {row['tolerance']})")
    print(f" RESULT: {'ALL CHECKS PASSED' if not failed else 'VERIFICATION FAILED'}")
    print(RULE)
    return 0 if (RESULTS and not failed) else 1


if __name__ == "__main__":
    sys.exit(main())
