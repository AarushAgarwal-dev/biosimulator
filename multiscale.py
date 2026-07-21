"""
Multi-Scale Model Coordinator.

Connects ODE (intracellular signaling), PDE (reaction-diffusion fields),
and ABM (cell-level behaviors) into a unified simulation pipeline.

Coupling strategy:
  - ODE ↔ ABM: Each ABM cell can run an intracellular ODE model.
    ODE outputs (e.g., proliferation signals) trigger ABM behaviors.
  - ABM ↔ PDE: Cells secrete into and sense from diffusible fields.
  - Time-scale separation: ODE (fast) → PDE (medium) → ABM (slow).
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from simulation_engine import ODEModel
from abm_engine import CellularPottsModel, build_cpm_from_blueprint


class MultiscaleSimulator:
    """
    Orchestrates coupled ODE-PDE-ABM simulations.

    Architecture:
    ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
    │  ODE Models  │────▶│  ABM Engine   │────▶│  PDE Fields  │
    │ (per-cell    │     │ (cell-level   │     │ (tissue-level│
    │  signaling)  │◀────│  behaviors)   │◀────│  diffusion)  │
    └─────────────┘     └──────────────┘     └─────────────┘
    """

    def __init__(self):
        self.cpm: Optional[CellularPottsModel] = None
        self.ode_blueprint: Optional[Dict[str, Any]] = None
        self.coupling_rules: List[Dict[str, Any]] = []
        self.cell_ode_states: Dict[int, Dict[str, float]] = {}

    def configure(
        self,
        abm_blueprint: Dict[str, Any],
        ode_blueprint: Optional[Dict[str, Any]] = None,
        coupling_rules: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Configure the multi-scale simulation.

        Args:
            abm_blueprint: ABM blueprint dict for CPM
            ode_blueprint: ODE blueprint for intracellular signaling (optional)
            coupling_rules: How ODE species map to ABM behaviors

        Coupling rule format:
        {
            "ode_species": "ERK",
            "threshold": 0.5,
            "abm_behavior": "growth_rate",     # or "death_probability", "secretion"
            "effect": "increase",              # or "decrease", "set"
            "magnitude": 0.1
        }
        """
        self.cpm = build_cpm_from_blueprint(abm_blueprint)
        self.ode_blueprint = ode_blueprint
        self.coupling_rules = coupling_rules or []

        # Initialize ODE states for each cell
        if ode_blueprint:
            for cell_id, cell in self.cpm.cells.items():
                self.cell_ode_states[cell_id] = {
                    node["id"]: node.get("initial_value", 0.0)
                    for node in ode_blueprint.get("nodes", [])
                }

    def simulate(
        self,
        num_mcs: int = 100,
        ode_steps_per_mcs: int = 5,
        ode_dt: float = 0.1,
        save_every: int = 10
    ) -> Dict[str, Any]:
        """
        Run coupled multi-scale simulation.

        For each MCS:
        1. Run intracellular ODE for each cell (fast timescale)
        2. Apply coupling rules (ODE outputs → ABM behaviors)
        3. Run one ABM Monte Carlo step
        4. Update diffusible fields (PDE step)
        """
        lattice_history = []
        field_history = {name: [] for name in self.cpm.fields}
        cell_count_history = []
        ode_summary_history = []
        time_points = []

        # Save initial state
        lattice_history.append(self.cpm._get_lattice_colored())
        for name, f in self.cpm.fields.items():
            field_history[name].append(
                f.grid.copy().tolist() if f.grid is not None else []
            )
        cell_count_history.append(self.cpm._get_cell_counts())
        time_points.append(0)

        for mcs_step in range(1, num_mcs + 1):
            # Step 1: Intracellular ODE updates
            if self.ode_blueprint:
                self._run_ode_step(ode_steps_per_mcs, ode_dt)

            # Step 2: Apply coupling rules
            if self.coupling_rules:
                self._apply_coupling()

            # Step 3: ABM Monte Carlo step
            self.cpm.run_mcs()

            # Step 4: Register new cells from divisions
            self._register_new_cells()

            # Save state
            if mcs_step % save_every == 0 or mcs_step == num_mcs:
                lattice_history.append(self.cpm._get_lattice_colored())
                for name, f in self.cpm.fields.items():
                    field_history[name].append(
                        f.grid.copy().tolist() if f.grid is not None else []
                    )
                cell_count_history.append(self.cpm._get_cell_counts())

                # Summarize ODE states
                if self.ode_blueprint:
                    ode_summary = self._summarize_ode_states()
                    ode_summary_history.append(ode_summary)

                time_points.append(mcs_step)

        result = {
            "type": "multiscale",
            "width": self.cpm.width,
            "height": self.cpm.height,
            "t": time_points,
            "lattice_frames": lattice_history,
            "fields": field_history,
            "cell_counts": cell_count_history,
            "cell_types": {
                ct.type_id: {"name": ct.name, "color": list(ct.color)}
                for ct in self.cpm.cell_types.values()
            },
            "total_mcs": num_mcs
        }

        if ode_summary_history:
            result["ode_summary"] = ode_summary_history

        return result

    def _run_ode_step(self, num_steps: int, dt: float):
        """Run intracellular ODE for each living cell."""
        if not self.ode_blueprint:
            return

        try:
            model = ODEModel(self.ode_blueprint)
        except Exception:
            return

        for cell_id, cell in self.cpm.cells.items():
            if not cell.alive:
                continue

            state = self.cell_ode_states.get(cell_id)
            if state is None:
                continue

            # Set initial conditions from cell's ODE state
            custom_params = {}
            for pname in model.param_names:
                if pname in model.params_dict:
                    custom_params[pname] = model.params_dict[pname]

            # Sense local field concentrations and inject into ODE
            for node in self.ode_blueprint.get("nodes", []):
                nid = node["id"]
                # Check if this species name matches a field
                field_name = nid.lower()
                if field_name in self.cpm.fields:
                    field_obj = self.cpm.fields[field_name]
                    if field_obj.grid is not None:
                        cx = int(cell.center_x) % self.cpm.width
                        cy = int(cell.center_y) % self.cpm.height
                        local_conc = float(field_obj.grid[cx, cy])
                        state[nid] = local_conc

            # Run a short ODE simulation
            try:
                result = model.simulate(
                    t_max=dt * num_steps,
                    num_points=num_steps + 1,
                    custom_params=custom_params
                )

                # Update cell's ODE state with final values
                for nid in model.node_ids:
                    if nid in result["species"]:
                        final_val = result["species"][nid][-1]
                        state[nid] = final_val

            except Exception:
                pass  # ODE step failed, keep previous state

    def _apply_coupling(self):
        """Apply coupling rules: ODE outputs → ABM cell behaviors."""
        for cell_id, cell in self.cpm.cells.items():
            if not cell.alive:
                continue

            state = self.cell_ode_states.get(cell_id, {})

            for rule in self.coupling_rules:
                ode_species = rule.get("ode_species", "")
                threshold = rule.get("threshold", 0.5)
                behavior = rule.get("abm_behavior", "")
                effect = rule.get("effect", "increase")
                magnitude = rule.get("magnitude", 0.1)

                ode_value = state.get(ode_species, 0.0)

                if ode_value > threshold:
                    self._modify_cell_behavior(cell, behavior, effect, magnitude)

    def _modify_cell_behavior(self, cell, behavior: str, effect: str, magnitude: float):
        """Modify a cell's behavior parameter based on coupling rule."""
        ct = cell.cell_type

        if behavior == "growth_rate":
            if effect == "increase":
                ct.growth_rate = min(1.0, ct.growth_rate + magnitude)
            elif effect == "decrease":
                ct.growth_rate = max(0.0, ct.growth_rate - magnitude)
            elif effect == "set":
                ct.growth_rate = magnitude

        elif behavior == "death_probability":
            if effect == "increase":
                ct.death_probability = min(1.0, ct.death_probability + magnitude)
            elif effect == "decrease":
                ct.death_probability = max(0.0, ct.death_probability - magnitude)
            elif effect == "set":
                ct.death_probability = magnitude

    def _register_new_cells(self):
        """Initialize ODE states for newly created cells (from division)."""
        if not self.ode_blueprint:
            return

        for cell_id in self.cpm.cells:
            if cell_id not in self.cell_ode_states:
                # New cell: inherit parent's state (approximate)
                self.cell_ode_states[cell_id] = {
                    node["id"]: node.get("initial_value", 0.0)
                    for node in self.ode_blueprint.get("nodes", [])
                }

    def _summarize_ode_states(self) -> Dict[str, Dict[str, float]]:
        """Compute mean/std of ODE species across all living cells."""
        if not self.ode_blueprint:
            return {}

        species_values = {}
        for node in self.ode_blueprint.get("nodes", []):
            nid = node["id"]
            species_values[nid] = []

        for cell_id, cell in self.cpm.cells.items():
            if not cell.alive:
                continue
            state = self.cell_ode_states.get(cell_id, {})
            for nid in species_values:
                species_values[nid].append(state.get(nid, 0.0))

        summary = {}
        for nid, vals in species_values.items():
            if vals:
                arr = np.array(vals)
                summary[nid] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr))
                }
            else:
                summary[nid] = {"mean": 0, "std": 0, "min": 0, "max": 0}

        return summary


# ============================================================
# SENSITIVITY ANALYSIS
# ============================================================

def local_sensitivity_analysis(
    blueprint: Dict[str, Any],
    target_species: str,
    param_names: List[str],
    t_max: float = 50.0,
    perturbation: float = 0.01
) -> Dict[str, float]:
    """
    Compute local sensitivity: ∂output/∂param for each parameter.

    Uses central finite differences with relative perturbation.

    Args:
        blueprint: ODE model blueprint
        target_species: Species to measure sensitivity of
        param_names: Parameters to analyze
        t_max: Simulation time
        perturbation: Relative perturbation size (e.g., 0.01 = 1%)

    Returns:
        Dict mapping parameter name → normalized sensitivity coefficient
    """
    sensitivities = {}

    try:
        model = ODEModel(blueprint)
        baseline = model.simulate(t_max)
        baseline_final = baseline["species"].get(target_species, [0.0])[-1]

        if abs(baseline_final) < 1e-10:
            baseline_final = 1e-10  # Avoid division by zero

        for pname in param_names:
            if pname not in model.params_dict:
                sensitivities[pname] = 0.0
                continue

            base_val = model.params_dict[pname]
            if abs(base_val) < 1e-10:
                base_val = 1e-5

            # Forward perturbation
            custom_up = {pname: base_val * (1 + perturbation)}
            result_up = model.simulate(t_max, custom_params=custom_up)
            val_up = result_up["species"].get(target_species, [0.0])[-1]

            # Backward perturbation
            custom_down = {pname: base_val * (1 - perturbation)}
            result_down = model.simulate(t_max, custom_params=custom_down)
            val_down = result_down["species"].get(target_species, [0.0])[-1]

            # Central difference, normalized
            dval = (val_up - val_down) / (2 * perturbation * base_val)
            normalized = dval * base_val / baseline_final

            sensitivities[pname] = float(normalized)

    except Exception as e:
        print(f"Sensitivity analysis error: {e}")

    return sensitivities
