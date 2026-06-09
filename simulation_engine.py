import numpy as np
import scipy.integrate
import scipy.optimize
import sympy as sp
from typing import Dict, List, Any, Tuple, Callable

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
        
        # Compile expressions
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
            
            # Basal synthesis rate
            synthesis_param = f"syn_{nid}"
            synthesis = self._get_param_symbol(synthesis_param, 0.0) # default 0 basal
            
            # Basal degradation rate
            deg_param = f"deg_{nid}"
            deg = self._get_param_symbol(deg_param, 0.1) # default degradation 0.1
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

    def get_equations_latex(self) -> Dict[str, str]:
        """
        Returns LaTeX representation of the equations for front-end rendering.
        """
        latex_eqs = {}
        for nid in self.node_ids:
            expr = self.deriv_exprs[nid]
            latex_expr = sp.latex(expr)
            latex_eqs[nid] = f"\\frac{{d[{nid}]}}{{dt}} = {latex_expr}"
        return latex_eqs

    def simulate(self, t_max: float, num_points: int = 100, custom_params: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Runs the simulation using scipy.integrate.solve_ivp
        """
        t_span = (0.0, t_max)
        t_eval = np.linspace(0.0, t_max, num_points)
        
        # Initial values vector
        y0 = [self.node_map[nid].get("initial_value", 0.0) for nid in self.node_ids]
        
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
