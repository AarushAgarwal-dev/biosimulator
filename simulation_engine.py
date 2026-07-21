import re
import numpy as np
import scipy.integrate
import scipy.optimize
import sympy as sp
from typing import Dict, List, Any, Tuple, Callable, Optional

try:
    from scipy.stats import qmc  # Quasi-Monte-Carlo (Latin Hypercube / Sobol)
    _HAS_QMC = True
except Exception:  # pragma: no cover - very old scipy fallback
    _HAS_QMC = False

# np.trapz was renamed to np.trapezoid in NumPy 2.0
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# ==========================================
# 1. ODE COMPILER & SOLVER
# ==========================================

class ODEModel:
    def __init__(self, blueprint: Dict[str, Any]):
        """
        Compiles a biological blueprint into a symbolic system of ODEs,
        then lambdifies it for fast numerical integration.
        """
        self.blueprint = blueprint
        self.nodes = blueprint.get("nodes", [])
        self.edges = blueprint.get("edges", [])
        
        # Sort nodes to have deterministic state indices
        self.node_ids = sorted([node["id"] for node in self.nodes])
        self.node_map = {n["id"]: n for n in self.nodes}
        
        # Symbolic variables
        self.t = sp.Symbol('t')
        self.vars = {nid: sp.Symbol(nid) for nid in self.node_ids}
        
        # Keep track of generated parameters
        self.params_dict = {}  # name -> default_value
        self.param_symbols = {}  # name -> sympy_symbol

        # Compile expressions. Two modes:
        #  - custom kinetics: explicit rate laws (blueprint["odes"] + optional
        #    shared fluxes) — reproduces exact mechanistic models (mass-conserving
        #    shared fluxes, arbitrary kinetics), like a published paper's ODE system.
        #  - generic: Hill activation/inhibition compiled from edges (default).
        if blueprint.get("odes"):
            self.deriv_exprs = self._compile_custom_system()
        else:
            self.deriv_exprs = self._compile_system()
        
        # Compile to a callable function
        # Signature: f(t, y_values, param_values)
        all_state_syms = [self.vars[nid] for nid in self.node_ids]
        all_param_names = sorted(self.params_dict.keys())
        all_param_syms = [self.param_symbols[name] for name in all_param_names]
        
        # Create lambdified derivative function
        self.param_names = all_param_names
        self._f_lambdified = sp.lambdify(
            (self.t, all_state_syms, all_param_syms),
            [self.deriv_exprs[nid] for nid in self.node_ids],
            'numpy'
        )

    def _get_param_symbol(self, name: str, default_val: float) -> sp.Symbol:
        if name not in self.param_symbols:
            self.param_symbols[name] = sp.Symbol(name)
            self.params_dict[name] = default_val
        return self.param_symbols[name]

    def _compile_system(self) -> Dict[str, sp.Expr]:
        derivs = {}
        
        # Group interactions by target
        incoming_edges = {nid: [] for nid in self.node_ids}
        for edge in self.edges:
            target = edge.get("target")
            if target in incoming_edges:
                incoming_edges[target].append(edge)
                
        for nid in self.node_ids:
            node = self.node_map[nid]
            initial_val = node.get("initial_value", 0.0)
            
            # Basal synthesis rate (carried on the node so tuned values persist)
            synthesis_param = f"syn_{nid}"
            synthesis = self._get_param_symbol(synthesis_param, float(node.get("synthesis", 0.0)))

            # Basal degradation rate (carried on the node so tuned values persist)
            deg_param = f"deg_{nid}"
            deg = self._get_param_symbol(deg_param, float(node.get("degradation", 0.1)))
            degradation = deg * self.vars[nid]
            
            # Group activators and inhibitors
            activators = []
            inhibitors = []
            for edge in incoming_edges[nid]:
                source = edge.get("source")
                etype = edge.get("type", "activation").lower()
                
                # Retrieve parameters or use defaults
                params = edge.get("parameters", {})
                k_val = params.get("k", 0.5)
                kd_val = params.get("K_d", 1.0)
                n_val = params.get("n", 2.0)
                
                if etype == "activation":
                    activators.append((source, k_val, kd_val, n_val))
                elif etype == "inhibition":
                    inhibitors.append((source, k_val, kd_val, n_val))
            
            # Construct activation term
            activation_expr = sp.Integer(0)
            if activators:
                for idx, (src, k, kd, n) in enumerate(activators):
                    # Symbols for parameters
                    k_sym = self._get_param_symbol(f"act_{src}_to_{nid}_k", k)
                    kd_sym = self._get_param_symbol(f"act_{src}_to_{nid}_Kd", kd)
                    n_sym = self._get_param_symbol(f"act_{src}_to_{nid}_n", n)
                    
                    # Hill activation term: k * src^n / (Kd^n + src^n)
                    src_sym = self.vars[src]
                    term = (k_sym * (src_sym ** n_sym)) / (kd_sym ** n_sym + src_sym ** n_sym)
                    activation_expr += term
            else:
                # If no activators, but there is basal synthesis, keep it. 
                # If there are inhibitors, they can inhibit basal synthesis.
                activation_expr = synthesis
                
            # Construct inhibition term (multiplier)
            inhibition_expr = sp.Integer(1)
            for idx, (src, k, kd, n) in enumerate(inhibitors):
                kd_sym = self._get_param_symbol(f"inh_{src}_to_{nid}_Kd", kd)
                n_sym = self._get_param_symbol(f"inh_{src}_to_{nid}_n", n)
                
                # Hill inhibition multiplier: Kd^n / (Kd^n + src^n)
                src_sym = self.vars[src]
                factor = (kd_sym ** n_sym) / (kd_sym ** n_sym + src_sym ** n_sym)
                inhibition_expr *= factor
                
            # Total derivative = synthesis + activation * inhibition - degradation
            if activators:
                # If there were activators, the synthesis is treated as additive
                total_production = synthesis + activation_expr * inhibition_expr
            else:
                total_production = activation_expr * inhibition_expr
                
            derivs[nid] = total_production - degradation

        return derivs

    def _compile_custom_system(self) -> Dict[str, sp.Expr]:
        """
        Build the ODE system from explicit rate laws (custom kinetics):

            blueprint["parameters"] : {name: value}          - rate constants etc.
            blueprint["fluxes"]     : {name: "expression"}    - named shared fluxes
            blueprint["odes"]       : {species_id: "d/dt expr"}

        Fluxes are substituted into the ODEs, so a single flux (e.g. v2, v3) can
        appear in several equations with opposite signs — exactly the mass-conserving
        structure the generic Hill compiler cannot express. All names are parsed with
        SymPy; species and declared parameters resolve to symbols.
        """
        params = self.blueprint.get("parameters", {}) or {}
        for pname, pval in params.items():
            try:
                self._get_param_symbol(str(pname), float(pval))
            except (TypeError, ValueError):
                self._get_param_symbol(str(pname), 0.0)

        # Symbol table: species + declared parameters (+ flux names as placeholders).
        local = {nid: self.vars[nid] for nid in self.node_ids}
        local.update({p: sym for p, sym in self.param_symbols.items()})

        fluxes = self.blueprint.get("fluxes", {}) or {}
        flux_syms = {f: sp.Symbol(f) for f in fluxes}
        local_with_flux = {**local, **flux_syms}

        flux_exprs = {f: sp.sympify(str(expr), locals=local_with_flux)
                      for f, expr in fluxes.items()}
        # Resolve any flux-referencing-flux by repeated substitution.
        for _ in range(len(flux_exprs) + 1):
            flux_exprs = {f: e.subs({flux_syms[g]: flux_exprs[g]
                                     for g in flux_exprs if g != f})
                          for f, e in flux_exprs.items()}

        odes = self.blueprint.get("odes", {}) or {}
        derivs: Dict[str, sp.Expr] = {}
        for nid in self.node_ids:
            expr = sp.sympify(str(odes.get(nid, "0")), locals=local_with_flux)
            expr = expr.subs({flux_syms[f]: flux_exprs[f] for f in flux_exprs})
            derivs[nid] = expr

        # Register any parameter used in the equations but not pre-declared, so the
        # lambdified function has a value for it (default from parameters, else 1.0).
        for nid in self.node_ids:
            for sym in derivs[nid].free_symbols:
                name = str(sym)
                if sym in self.vars.values() or name in self.param_symbols or sym == self.t:
                    continue
                self._get_param_symbol(name, float(params.get(name, 1.0)))

        return derivs

    def _verbose_param_latex(self, pname: str) -> str:
        """Map a compact parameter name (e.g. act_EGFR_to_CBL_k) to a spelled-out LaTeX label."""
        # Underscores are literal in a parameter name but are subscript operators in
        # LaTeX/KaTeX, and are illegal inside \text{}/\mathrm{}. Escape them so
        # custom-kinetics names like t_on, t_off, K_d render instead of erroring.
        def esc(s):
            return str(s).replace("\\", r"\backslash ").replace("_", r"\_")
        if pname.startswith("deg_"):
            return f"\\text{{degradation}}_{{\\mathrm{{{esc(pname[4:])}}}}}"
        if pname.startswith("syn_"):
            return f"\\text{{synthesis}}_{{\\mathrm{{{esc(pname[4:])}}}}}"
        m = re.match(r"^(act|inh)_(.+)_to_(.+)_(k|Kd|n)$", pname)
        if m:
            kind, src, tgt, suffix = m.groups()
            label = {"k": "activation strength",
                     "Kd": "half-saturation",
                     "n": "Hill coefficient"}[suffix]
            edge = f"\\mathrm{{{esc(src)}}}\\!\\to\\!\\mathrm{{{esc(tgt)}}}"
            sup = "" if kind == "act" else "^{\\text{inh}}"
            return f"\\text{{{label}}}{sup}_{{{edge}}}"
        return f"\\text{{{esc(pname)}}}"

    def get_equations_latex(self, verbose: bool = False) -> Dict[str, str]:
        """
        Returns LaTeX representation of the equations for front-end rendering.

        verbose=True spells out every parameter (degradation, synthesis,
        activation strength, half-saturation, Hill coefficient) and uses each
        species' full descriptive name instead of its short id.
        """
        symbol_names = {}
        if verbose:
            for nid in self.node_ids:
                full = self.node_map[nid].get("name") or nid
                symbol_names[self.vars[nid]] = f"[\\text{{{full}}}]"
            for pname, sym in self.param_symbols.items():
                symbol_names[sym] = self._verbose_param_latex(pname)

        latex_eqs = {}
        for nid in self.node_ids:
            expr = self.deriv_exprs[nid]
            if verbose:
                latex_expr = sp.latex(expr, symbol_names=symbol_names)
                lhs = self.node_map[nid].get("name") or nid
                latex_eqs[nid] = f"\\frac{{d[\\text{{{lhs}}}]}}{{dt}} = {latex_expr}"
            else:
                latex_expr = sp.latex(expr)
                latex_eqs[nid] = f"\\frac{{d[{nid}]}}{{dt}} = {latex_expr}"
        return latex_eqs

    def simulate(self, t_max: float, num_points: int = 100, custom_params: Dict[str, float] = None,
                 custom_initial: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Runs the simulation using scipy.integrate.solve_ivp.
        custom_initial overrides specific species' initial values (used by the
        multi-condition tests, e.g. bistability from a low vs. high start).
        """
        t_span = (0.0, t_max)
        t_eval = np.linspace(0.0, t_max, num_points)

        # Initial values vector (with optional per-species overrides)
        y0 = []
        for nid in self.node_ids:
            if custom_initial and nid in custom_initial:
                y0.append(custom_initial[nid])
            else:
                y0.append(self.node_map[nid].get("initial_value", 0.0))
        
        # Parameter values vector
        param_vals = []
        for name in self.param_names:
            if custom_params and name in custom_params:
                param_vals.append(custom_params[name])
            else:
                param_vals.append(self.params_dict[name])
                
        def rhs(t, y):
            return self._f_lambdified(t, y, param_vals)
            
        sol = scipy.integrate.solve_ivp(rhs, t_span, y0, t_eval=t_eval, method='RK45')
        
        # Format results
        results = {
            "t": sol.t.tolist(),
            "species": {}
        }
        for idx, nid in enumerate(self.node_ids):
            results["species"][nid] = sol.y[idx].tolist()
            
        return results

    def fit_parameters_to_target(self, target_data: Dict[str, List[float]], target_times: List[float], params_to_fit: List[str]) -> Tuple[Dict[str, float], float]:
        """
        Fits selected parameters to experimental target curves using Scipy least-squares optimization.
        """
        if not params_to_fit:
            return {}, 0.0
            
        # Initial parameter guess
        x0 = [self.params_dict[p] for p in params_to_fit]
        
        # Setup target vector
        target_species = list(target_data.keys())
        target_y = []
        for sp_name in target_species:
            target_y.extend(target_data[sp_name])
        target_y = np.array(target_y)
        
        # Optimization loss function
        def objective(x):
            # Create dict of current parameter values
            curr_params = self.params_dict.copy()
            for idx, p_name in enumerate(params_to_fit):
                # Restrict parameters to be non-negative
                curr_params[p_name] = max(1e-5, x[idx])
                
            # Simulate at the target time points
            t_span = (0.0, max(target_times))
            y0 = [self.node_map[nid].get("initial_value", 0.0) for nid in self.node_ids]
            
            param_vals = [curr_params[name] for name in self.param_names]
            
            def rhs(t, y):
                return self._f_lambdified(t, y, param_vals)
                
            sol = scipy.integrate.solve_ivp(rhs, t_span, y0, t_eval=target_times, method='RK45')
            
            if not sol.success:
                return 1e6 * np.ones_like(target_y)
                
            # Build current simulation output vector
            sim_y = []
            for sp_name in target_species:
                idx = self.node_ids.index(sp_name)
                sim_y.extend(sol.y[idx])
            sim_y = np.array(sim_y)
            
            # Return residuals
            return sim_y - target_y
            
        # Bounds: parameters must be positive
        bounds = (0.0, 100.0)
        res = scipy.optimize.least_squares(objective, x0, bounds=bounds)
        
        fitted_params = {}
        for idx, p_name in enumerate(params_to_fit):
            fitted_params[p_name] = float(res.x[idx])
            
        return fitted_params, float(np.sum(res.fun**2))


# ==========================================
# 2. PDE SOLVER (REACTION-DIFFUSION)
# ==========================================

def solve_pde(
    spatial_config: Dict[str, Any],
    reaction_formulas: Dict[str, str],
    initial_conditions: Dict[str, Any],
    t_max: float,
    dt: float = 0.1,
    save_every: int = 10
) -> Dict[str, Any]:
    """
    Solves a 2D reaction-diffusion system using finite difference schemes:
    u_t = D_u * del^2 u + f(u, v)
    Boundary conditions: Zero-flux (Neumann)
    """
    Nx = spatial_config.get("x_grid", 50)
    Ny = spatial_config.get("y_grid", 50)
    dx = spatial_config.get("dx", 1.0)
    dy = spatial_config.get("dy", 1.0)
    
    species_names = list(reaction_formulas.keys())
    diffusions = spatial_config.get("diffusion", {})
    
    # Compile reaction terms symbolically using SymPy
    symbols = {name: sp.Symbol(name) for name in species_names}
    lambdas = {}
    for name, formula in reaction_formulas.items():
        expr = sp.sympify(formula)
        # compile formula to take values for each species
        lambdas[name] = sp.lambdify(
            [symbols[sp_name] for sp_name in species_names],
            expr,
            'numpy'
        )
        
    # Initialize grids
    grids = {}
    for name in species_names:
        init_type = initial_conditions.get(name, {}).get("type", "random_noise")
        base_val = initial_conditions.get(name, {}).get("base_value", 1.0)
        
        if init_type == "random_noise":
            noise_amp = initial_conditions.get(name, {}).get("noise_amplitude", 0.05)
            # Random perturbations around base value
            grids[name] = base_val + noise_amp * (np.random.rand(Nx, Ny) - 0.5)
        elif init_type == "central_spot":
            spot_radius = initial_conditions.get(name, {}).get("spot_radius", 5)
            spot_val = initial_conditions.get(name, {}).get("spot_value", 2.0)
            grid = np.ones((Nx, Ny)) * base_val
            cx, cy = Nx // 2, Ny // 2
            for i in range(Nx):
                for j in range(Ny):
                    if (i - cx)**2 + (j - cy)**2 <= spot_radius**2:
                        grid[i, j] = spot_val
            grids[name] = grid
        else:
            grids[name] = np.ones((Nx, Ny)) * base_val
            
    # Simulation loop
    steps = int(t_max / dt)
    history = {name: [] for name in species_names}
    time_points = []
    
    # Save initial state
    for name in species_names:
        history[name].append(grids[name].copy().tolist())
    time_points.append(0.0)
    
    # Laplace operator helper (zero-flux Neumann boundary conditions)
    def compute_laplacian(arr, dx, dy):
        lap = np.zeros_like(arr)
        # Stencil for interior points
        lap[1:-1, 1:-1] = (
            (arr[2:, 1:-1] - 2 * arr[1:-1, 1:-1] + arr[:-2, 1:-1]) / (dx**2) +
            (arr[1:-1, 2:] - 2 * arr[1:-1, 1:-1] + arr[1:-1, :-2]) / (dy**2)
        )
        # Apply Neumann Boundary conditions (zero gradient on borders)
        # Top & Bottom rows
        arr_padded = np.pad(arr, 1, mode='edge')
        lap_padded = (
            (arr_padded[2:, 1:-1] - 2 * arr_padded[1:-1, 1:-1] + arr_padded[:-2, 1:-1]) / (dx**2) +
            (arr_padded[1:-1, 2:] - 2 * arr_padded[1:-1, 1:-1] + arr_padded[1:-1, :-2]) / (dy**2)
        )
        return lap_padded
        
    curr_grids = {name: grids[name].copy() for name in species_names}
    
    for step in range(1, steps + 1):
        next_grids = {}
        # Fetch current grid arrays
        species_values = [curr_grids[name] for name in species_names]
        
        # Calculate reactions for all species
        reactions = {}
        for name in species_names:
            # Evaluate compiled lambda function element-wise
            reactions[name] = lambdas[name](*species_values)
            
        for name in species_names:
            D = diffusions.get(name, 0.1)
            lap = compute_laplacian(curr_grids[name], dx, dy)
            
            # Euler time step: dU/dt = D*laplacian + reaction
            next_grids[name] = curr_grids[name] + dt * (D * lap + reactions[name])
            
            # Keep boundaries non-negative
            next_grids[name] = np.maximum(next_grids[name], 0.0)
            
        curr_grids = next_grids
        
        # Save frame
        if step % save_every == 0 or step == steps:
            for name in species_names:
                history[name].append(curr_grids[name].copy().tolist())
            time_points.append(step * dt)
            
    return {
        "t": time_points,
        "x_size": Nx,
        "y_size": Ny,
        "species": history
    }


# ==========================================
# 3. PARAMETER-SPACE EXPLORATION (SAMPLING)
# ==========================================

def _unit_samples(method: str, dims: int, n_samples: int, seed: int = 0) -> np.ndarray:
    """
    Draw `n_samples` points in the unit hypercube [0, 1]^dims using the
    requested sampling strategy. Falls back gracefully if scipy.stats.qmc
    is unavailable.
    """
    method = (method or "lhs").lower()
    rng = np.random.default_rng(seed)

    if _HAS_QMC and method in ("lhs", "latin", "latin_hypercube"):
        sampler = qmc.LatinHypercube(d=dims, seed=seed)
        return sampler.random(n=n_samples)
    if _HAS_QMC and method in ("sobol", "quasi"):
        sampler = qmc.Sobol(d=dims, scramble=True, seed=seed)
        # Sobol is happiest with power-of-two counts, but random() handles any n.
        return sampler.random(n=n_samples)
    if method in ("grid", "factorial"):
        # Even factorial grid; number of points per axis chosen so the total
        # is close to (but not more than) n_samples.
        per_axis = max(2, int(round(n_samples ** (1.0 / max(1, dims)))))
        axes = [np.linspace(0.0, 1.0, per_axis) for _ in range(dims)]
        mesh = np.meshgrid(*axes, indexing="ij")
        grid = np.stack([m.ravel() for m in mesh], axis=-1)
        if grid.shape[0] > n_samples:
            idx = np.linspace(0, grid.shape[0] - 1, n_samples).astype(int)
            grid = grid[idx]
        return grid

    # Plain Monte-Carlo fallback (also the manual-LHS path when qmc missing)
    if not _HAS_QMC and method in ("lhs", "latin", "latin_hypercube"):
        # Manual Latin Hypercube: one stratified draw per axis, shuffled.
        cut = np.linspace(0.0, 1.0, n_samples + 1)
        u = rng.uniform(size=(n_samples, dims))
        pts = cut[:n_samples, None] + u * (1.0 / n_samples)
        for j in range(dims):
            rng.shuffle(pts[:, j])
        return pts
    return rng.uniform(size=(n_samples, dims))


def _scale_samples(unit: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> np.ndarray:
    """Scale unit-cube samples into [low, high] per dimension (safe for low==high)."""
    span = highs - lows
    # Degenerate axes (min == max) stay pinned at the shared value.
    span = np.where(span <= 0, 0.0, span)
    return lows + unit * span


def _trajectory_metrics(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Summary statistics for a single species time-course."""
    if y.size == 0:
        return {"peak_value": 0.0, "peak_time": 0.0, "final_value": 0.0, "auc": 0.0}
    peak_idx = int(np.argmax(y))
    return {
        "peak_value": float(y[peak_idx]),
        "peak_time": float(t[peak_idx]),
        "final_value": float(y[-1]),
        "auc": float(_trapz(y, t)) if _trapz else float(np.sum(y)),
    }


def explore_parameter_space(
    blueprint: Dict[str, Any],
    param_bounds: Dict[str, Dict[str, float]],
    n_samples: int = 64,
    method: str = "lhs",
    target_species: Optional[str] = None,
    t_max: Optional[float] = None,
    seed: int = 0,
    max_trajectories: int = 60,
) -> Dict[str, Any]:
    """
    Explore a model's parameter space by sampling the requested parameters
    (Latin Hypercube by default) inside user-supplied [min, max] bounds,
    running a simulation for each sample, and summarising the outcome for a
    chosen target species.

    param_bounds: {param_name: {"min": float, "max": float}}
    Returns per-sample parameter sets, output metrics, and (for the first
    `max_trajectories` samples) the target-species trajectories for an
    ensemble plot.
    """
    model = ODEModel(blueprint)

    # Only sample parameters the model actually knows about.
    names = sorted(p for p in param_bounds.keys() if p in model.params_dict)
    if not names:
        raise ValueError("None of the requested parameters exist in this model.")

    n_samples = int(max(1, min(n_samples, 2000)))
    lows = np.array([float(param_bounds[p].get("min", 0.0)) for p in names], dtype=float)
    highs = np.array([float(param_bounds[p].get("max", 1.0)) for p in names], dtype=float)
    # Repair inverted bounds instead of failing.
    swap = highs < lows
    lows[swap], highs[swap] = highs[swap], lows[swap]

    unit = _unit_samples(method, len(names), n_samples, seed)
    scaled = _scale_samples(unit, lows, highs)
    actual_n = scaled.shape[0]

    # Simulation horizon + evaluation grid (shared across samples).
    if t_max is None:
        t_max = float(blueprint.get("simulation_config", {}).get("t_max", 50.0))
    num_points = 100
    t_eval = np.linspace(0.0, t_max, num_points)
    t_span = (0.0, t_max)
    y0 = [model.node_map[nid].get("initial_value", 0.0) for nid in model.node_ids]

    # Choose the species we report metrics/trajectories for.
    if target_species not in model.node_ids:
        target_species = model.node_ids[-1] if model.node_ids else None
    tgt_idx = model.node_ids.index(target_species) if target_species else 0

    samples_out: List[Dict[str, Any]] = []
    failures = 0

    for i in range(actual_n):
        # Build the parameter vector: sampled values override defaults.
        overrides = {names[j]: float(scaled[i, j]) for j in range(len(names))}
        param_vals = [
            overrides.get(name, model.params_dict[name]) for name in model.param_names
        ]

        def rhs(t, y, _pv=param_vals):
            return model._f_lambdified(t, y, _pv)

        try:
            sol = scipy.integrate.solve_ivp(
                rhs, t_span, y0, t_eval=t_eval, method="RK45"
            )
            ok = bool(sol.success)
        except Exception:
            ok = False

        if not ok:
            failures += 1
            samples_out.append({
                "id": i,
                "params": overrides,
                "ok": False,
                "metrics": {"peak_value": 0.0, "peak_time": 0.0, "final_value": 0.0, "auc": 0.0},
            })
            continue

        y_tgt = np.asarray(sol.y[tgt_idx], dtype=float)
        metrics = _trajectory_metrics(sol.t, y_tgt)
        entry = {
            "id": i,
            "params": overrides,
            "ok": True,
            "metrics": metrics,
        }
        if i < max_trajectories:
            # Round to keep the JSON payload light.
            entry["trajectory"] = [round(float(v), 4) for v in y_tgt]
        samples_out.append(entry)

    # Aggregate ranges for quick summary display.
    ok_samples = [s for s in samples_out if s["ok"]]
    def _range(key):
        vals = [s["metrics"][key] for s in ok_samples]
        return {"min": float(min(vals)), "max": float(max(vals)),
                "mean": float(np.mean(vals))} if vals else {"min": 0, "max": 0, "mean": 0}

    return {
        "method": method,
        "param_names": names,
        "bounds": {names[j]: {"min": float(lows[j]), "max": float(highs[j])}
                   for j in range(len(names))},
        "target_species": target_species,
        "t": [round(float(v), 4) for v in t_eval],
        "n_requested": n_samples,
        "n_evaluated": actual_n,
        "n_failed": failures,
        "samples": samples_out,
        "metric_ranges": {
            "peak_value": _range("peak_value"),
            "peak_time": _range("peak_time"),
            "final_value": _range("final_value"),
            "auc": _range("auc"),
        },
    }
