"""
1D reaction-diffusion-advection solver with real boundary conditions.

Solves, on a uniform 1D mesh,

    du/dt = D d2u/dx2 - v du/dx + R(u, x, t) + S(x, t)

with Dirichlet, Neumann (including no-flux), Robin or periodic ends.

Why this exists alongside ``simulation_engine.solve_pde``
--------------------------------------------------------
``solve_pde`` is 2D and hard-wires zero-flux boundaries. A 1D model with a FIXED
value at each end is the canonical verification case -- its steady state is an exact
straight line -- and that cannot be expressed with Neumann-only boundaries. Rather
than bend the 2D solver, this module implements the 1D case directly and shares the
same conventions: explicit time stepping, a stability check that refuses rather than
diverges, and a reported mass series.

Numerical assumptions and limitations
-------------------------------------
* Time integration is explicit forward Euler: first-order and conditionally stable.
  The step must satisfy BOTH the diffusive limit ``dt <= dx^2/(2D)`` and the
  advective CFL ``dt <= dx/|v|``; the combined limit is enforced.
* Advection uses first-order UPWIND differencing. It is monotone (no spurious
  oscillation) at the cost of numerical diffusion of order ``|v| dx / 2``, which is
  reported in the returned ``stability`` block so the smearing is not mistaken for
  physical diffusion.
* Diffusion is constant in space. A spatially varying D would need the flux form
  ``d/dx(D du/dx)``, which is not implemented.
* Dirichlet ends are imposed after each step; the reported mass therefore is NOT
  conserved at a Dirichlet boundary, which is correct -- material enters or leaves.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


def solve_1d(
    x: Sequence[float],
    initial: Sequence[float],
    diffusion: float,
    t_max: float,
    dt: float,
    reaction: Optional[Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = None,
    source: Optional[Callable[[np.ndarray, float], np.ndarray]] = None,
    advection: float = 0.0,
    left: Optional[Dict[str, Any]] = None,
    right: Optional[Dict[str, Any]] = None,
    periodic: bool = False,
    save_every: int = 1,
    strict_stability: bool = True,
) -> Dict[str, Any]:
    """Integrate the 1D problem and return frames, times, mass and diagnostics."""
    nodes = np.asarray(x, dtype=float)
    u = np.array(initial, dtype=float)
    if nodes.ndim != 1 or nodes.size < 3:
        raise ValueError("A 1D mesh needs at least 3 nodes.")
    if u.shape != nodes.shape:
        raise ValueError(f"Initial values have length {u.size}, but the mesh has "
                         f"{nodes.size} nodes.")
    spacing = np.diff(nodes)
    if np.any(spacing <= 0):
        raise ValueError("Mesh nodes must be strictly increasing.")
    dx = float(spacing[0])
    if not np.allclose(spacing, dx, rtol=1e-9, atol=1e-12):
        raise ValueError("solve_1d requires a uniform mesh; spacing varies.")

    D = float(diffusion)
    v = float(advection)
    if D < 0:
        raise ValueError("The diffusion coefficient cannot be negative.")

    # Combined explicit stability limit.
    diffusive_limit = (dx * dx / (2.0 * D)) if D > 0 else float("inf")
    advective_limit = (dx / abs(v)) if abs(v) > 0 else float("inf")
    dt_max = min(diffusive_limit, advective_limit)
    if strict_stability and dt > dt_max * (1.0 + 1e-12):
        raise ValueError(
            f"The explicit scheme is unstable for these settings: dt={dt:g} exceeds the "
            f"limit {dt_max:.4g} (diffusive limit {diffusive_limit:.4g}, advective limit "
            f"{advective_limit:.4g}) at spacing dx={dx:g}. Reduce dt, coarsen the mesh, or "
            f"reduce D / the velocity."
        )

    left = dict(left or {"type": "no_flux", "flux": 0.0})
    right = dict(right or {"type": "no_flux", "flux": 0.0})
    if periodic:
        left = {"type": "periodic"}
        right = {"type": "periodic"}

    # Time grid: index-derived so long runs do not drift, last stamp pinned to t_max.
    n_full = int(np.floor(t_max / dt + 1e-9))
    remainder = t_max - n_full * dt
    step_sizes = [dt] * n_full
    step_times = [dt * i for i in range(1, n_full + 1)]
    if remainder > 1e-12 * max(1.0, t_max):
        step_sizes.append(remainder)
        step_times.append(t_max)
    elif step_times:
        step_times[-1] = t_max
    steps = len(step_sizes)

    frames: List[List[float]] = []
    times: List[float] = []
    mass: List[float] = []

    def apply_dirichlet(values: np.ndarray) -> None:
        if left.get("type") == "dirichlet":
            values[0] = float(left["value"])
        if right.get("type") == "dirichlet":
            values[-1] = float(right["value"])

    def record(values: np.ndarray, t: float) -> None:
        frames.append([float(value) for value in values])
        times.append(float(t))
        # Trapezoidal integral: the correct discrete mass on a node-centred mesh.
        mass.append(float(np.trapezoid(values, nodes) if hasattr(np, "trapezoid")
                          else np.trapz(values, nodes)))

    apply_dirichlet(u)
    record(u, 0.0)

    def ghost_values(values: np.ndarray) -> tuple:
        """Ghost nodes implementing the flux/periodic boundary conditions.

        For a Neumann condition -D du/dn = q with outward normal n, the one-sided
        difference gives the ghost value directly; Robin substitutes
        q = h (u_boundary - u_inf).
        """
        if periodic:
            return values[-2], values[1]

        def ghost(end: Dict[str, Any], boundary_value: float, inward: float) -> float:
            kind = end.get("type")
            if kind == "dirichlet":
                # Mirror so the second derivative at the fixed node is well defined;
                # the node itself is overwritten by apply_dirichlet afterwards.
                return 2.0 * float(end["value"]) - inward
            if kind == "robin":
                h = float(end.get("transfer_coefficient", 0.0))
                u_inf = float(end.get("ambient_value", 0.0))
                q = h * (boundary_value - u_inf)
            else:                                   # neumann / no_flux
                q = float(end.get("flux", 0.0))
            # -D du/dn = q  =>  du/dn = -q/D. With n pointing out of the domain,
            # (u_ghost - u_inward) / (2 dx) = -q/D.
            if D <= 0:
                return boundary_value
            return inward - 2.0 * dx * q / D

        return (ghost(left, values[0], values[1]),
                ghost(right, values[-1], values[-2]))

    elapsed = 0.0
    diverged_at: Optional[float] = None

    for step, (h, t_now) in enumerate(zip(step_sizes, step_times), start=1):
        left_ghost, right_ghost = ghost_values(u)
        extended = np.concatenate(([left_ghost], u, [right_ghost]))

        # Second-order centred diffusion.
        laplacian = (extended[2:] - 2.0 * extended[1:-1] + extended[:-2]) / (dx * dx)

        # First-order upwind advection: take the difference from the direction the
        # flow comes FROM, which is what makes the scheme monotone.
        if v > 0:
            gradient = (extended[1:-1] - extended[:-2]) / dx
        elif v < 0:
            gradient = (extended[2:] - extended[1:-1]) / dx
        else:
            gradient = np.zeros_like(u)

        rhs = D * laplacian - v * gradient
        if reaction is not None:
            rhs = rhs + np.asarray(reaction(u, nodes, elapsed), dtype=float)
        if source is not None:
            rhs = rhs + np.asarray(source(nodes, elapsed), dtype=float)

        u = u + h * rhs
        apply_dirichlet(u)
        elapsed = t_now

        if not np.all(np.isfinite(u)):
            diverged_at = elapsed
            record(u, elapsed)
            break
        if step % save_every == 0 or step == steps:
            record(u, elapsed)

    return {
        "x": [float(value) for value in nodes],
        "t": times,
        "u": frames,
        "mass": mass,
        "stability": {
            "dt": float(dt),
            "dx": float(dx),
            "max_stable_dt": float(dt_max) if np.isfinite(dt_max) else None,
            "diffusive_limit": float(diffusive_limit) if np.isfinite(diffusive_limit) else None,
            "advective_limit": float(advective_limit) if np.isfinite(advective_limit) else None,
            "diffusion_number": float(dt * D / (dx * dx)) if dx > 0 else None,
            "courant_number": float(dt * abs(v) / dx) if dx > 0 else None,
            # Upwind differencing adds this much artificial diffusion. Reported so a
            # smeared front is attributed to the scheme, not to physics.
            "numerical_diffusion": float(abs(v) * dx / 2.0),
            "steps": steps,
            "end_time": float(elapsed),
            "diverged_at": diverged_at,
        },
        "boundaries": {"left": left, "right": right, "periodic": bool(periodic)},
    }


def steady_state_1d(x: Sequence[float], diffusion: float,
                    left_value: float, right_value: float) -> np.ndarray:
    """Analytic steady state for pure diffusion between two fixed values.

    With no reaction and no advection, d2u/dx2 = 0, so the profile is the straight
    line joining the two boundary values. Used to verify the solver rather than to
    produce results.
    """
    nodes = np.asarray(x, dtype=float)
    span = nodes[-1] - nodes[0]
    if span <= 0:
        raise ValueError("The mesh has no extent.")
    return left_value + (right_value - left_value) * (nodes - nodes[0]) / span
