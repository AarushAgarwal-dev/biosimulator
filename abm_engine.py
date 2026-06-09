"""
Full Cellular Potts Model (CPM/GGH) Agent-Based Modeling Engine.

Inspired by CompuCell3D: implements a Glazier-Graner-Hogeweg model with
Monte Carlo Metropolis dynamics, Hamiltonian energy minimization, cell
behaviors (division, death, secretion, chemotaxis), and coupled
reaction-diffusion fields.
"""

import numpy as np
import math
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# CELL & CELL TYPE DEFINITIONS
# ============================================================

@dataclass
class CellType:
    """Definition of a cell type with its physical and behavioral properties."""
    type_id: int
    name: str
    target_volume: float = 25.0        # Target volume in pixels
    lambda_volume: float = 2.0          # Volume constraint strength
    target_surface: float = 20.0        # Target surface area in pixels
    lambda_surface: float = 1.0         # Surface constraint strength
    max_volume_before_division: float = 50.0  # Trigger mitosis above this
    growth_rate: float = 0.0            # Volume increase per MCS (Monte Carlo Step)
    death_probability: float = 0.0      # Probability of death per MCS
    chemotaxis_lambda: float = 0.0      # Chemotaxis strength
    chemotaxis_field: str = ""          # Which field to follow
    secretion_rates: Dict[str, float] = field(default_factory=dict)  # field_name -> rate
    uptake_rates: Dict[str, float] = field(default_factory=dict)     # field_name -> rate
    color: Tuple[int, int, int] = (100, 100, 255)  # RGB display color
    freeze: bool = False                # Frozen cells don't update


@dataclass
class Cell:
    """A single cell in the simulation."""
    cell_id: int
    cell_type: CellType
    volume: int = 0
    surface: int = 0
    center_x: float = 0.0
    center_y: float = 0.0
    internal_state: Dict[str, float] = field(default_factory=dict)
    age: int = 0  # MCS since creation
    alive: bool = True


# ============================================================
# DIFFUSIBLE FIELD
# ============================================================

@dataclass
class DiffusibleField:
    """A reaction-diffusion field coupled to the cell lattice."""
    name: str
    diffusion_coefficient: float = 0.1
    decay_rate: float = 0.01
    grid: Optional[np.ndarray] = None
    boundary_condition: str = "neumann"  # "neumann" or "periodic"

    def initialize(self, width: int, height: int, initial_value: float = 0.0):
        self.grid = np.ones((width, height)) * initial_value

    def diffuse_and_decay(self, dt: float = 1.0):
        """One step of diffusion + decay using finite differences."""
        if self.grid is None:
            return

        D = self.diffusion_coefficient
        k = self.decay_rate
        g = self.grid

        # Compute Laplacian with boundary handling
        if self.boundary_condition == "periodic":
            lap = (
                np.roll(g, 1, axis=0) + np.roll(g, -1, axis=0) +
                np.roll(g, 1, axis=1) + np.roll(g, -1, axis=1) - 4 * g
            )
        else:
            # Neumann (zero-flux) via padding
            padded = np.pad(g, 1, mode='edge')
            lap = (
                padded[2:, 1:-1] + padded[:-2, 1:-1] +
                padded[1:-1, 2:] + padded[1:-1, :-2] - 4 * padded[1:-1, 1:-1]
            )

        # Euler step
        self.grid = g + dt * (D * lap - k * g)
        self.grid = np.maximum(self.grid, 0.0)


# ============================================================
# CELLULAR POTTS MODEL ENGINE
# ============================================================

class CellularPottsModel:
    """
    Full Cellular Potts Model with Hamiltonian energy minimization.

    The Hamiltonian:
      H = Σ_adhesion J(τ_i, τ_j) * (1 - δ(σ_i, σ_j))
        + Σ_cells λ_vol * (v - V_target)²
        + Σ_cells λ_surf * (s - S_target)²
        - Σ_cells λ_chem * (c(x_target) - c(x_source))

    Monte Carlo Metropolis dynamics: pixel copy attempts with Boltzmann acceptance.
    """

    def __init__(
        self,
        width: int = 100,
        height: int = 100,
        temperature: float = 10.0,
        neighbor_order: int = 1
    ):
        self.width = width
        self.height = height
        self.temperature = temperature
        self.neighbor_order = neighbor_order

        # Lattice: each pixel stores a cell ID (0 = medium)
        self.lattice = np.zeros((width, height), dtype=np.int32)

        # Cell registry
        self.cells: Dict[int, Cell] = {}
        self.cell_types: Dict[int, CellType] = {}
        self.next_cell_id = 1

        # Adhesion energy matrix: J[type_i][type_j]
        # type_id 0 = medium
        self.adhesion_matrix: Dict[Tuple[int, int], float] = {}

        # Diffusible fields
        self.fields: Dict[str, DiffusibleField] = {}

        # Simulation state
        self.mcs = 0  # Monte Carlo step counter
        self.history: List[Dict[str, Any]] = []

        # Pre-compute neighbor offsets
        self._neighbor_offsets = self._compute_neighbor_offsets()

        # Register medium as type 0
        self.cell_types[0] = CellType(type_id=0, name="Medium", color=(20, 20, 30))

    def _compute_neighbor_offsets(self) -> List[Tuple[int, int]]:
        """Compute neighbor pixel offsets based on neighbor_order."""
        offsets = []
        r = self.neighbor_order
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue
                if abs(dx) + abs(dy) <= r:  # Manhattan distance
                    offsets.append((dx, dy))
        return offsets

    # ----------------------------------------------------------
    # CELL TYPE & CELL MANAGEMENT
    # ----------------------------------------------------------

    def register_cell_type(self, cell_type: CellType):
        """Register a cell type definition."""
        self.cell_types[cell_type.type_id] = cell_type

    def set_adhesion(self, type_a: int, type_b: int, energy: float):
        """Set adhesion energy between two cell types (symmetric)."""
        self.adhesion_matrix[(type_a, type_b)] = energy
        self.adhesion_matrix[(type_b, type_a)] = energy

    def get_adhesion(self, type_a: int, type_b: int) -> float:
        """Get adhesion energy. Default = 16.0 if not set."""
        return self.adhesion_matrix.get((type_a, type_b), 16.0)

    def create_cell(self, cell_type_id: int) -> Cell:
        """Create a new cell of the given type."""
        ctype = self.cell_types[cell_type_id]
        cell = Cell(
            cell_id=self.next_cell_id,
            cell_type=ctype,
        )
        self.cells[self.next_cell_id] = cell
        self.next_cell_id += 1
        return cell

    def seed_cell_at(self, cell: Cell, cx: int, cy: int, radius: int = 3):
        """Place a cell as a disk centered at (cx, cy) with given radius."""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x = (cx + dx) % self.width
                    y = (cy + dy) % self.height
                    self.lattice[x, y] = cell.cell_id
        self._update_cell_stats(cell)

    def seed_random_cells(self, cell_type_id: int, count: int, radius: int = 3):
        """Seed `count` cells of given type at random positions."""
        for _ in range(count):
            cell = self.create_cell(cell_type_id)
            cx = random.randint(radius, self.width - radius - 1)
            cy = random.randint(radius, self.height - radius - 1)
            self.seed_cell_at(cell, cx, cy, radius)

    def seed_block(self, cell_type_id: int, x0: int, y0: int, x1: int, y1: int, cell_size: int = 5):
        """Fill a rectangular region with cells of given type in a grid pattern."""
        for x in range(x0, x1, cell_size):
            for y in range(y0, y1, cell_size):
                cell = self.create_cell(cell_type_id)
                # Fill a small square
                for dx in range(min(cell_size, x1 - x)):
                    for dy in range(min(cell_size, y1 - y)):
                        px = min(x + dx, self.width - 1)
                        py = min(y + dy, self.height - 1)
                        self.lattice[px, py] = cell.cell_id
                self._update_cell_stats(cell)

    # ----------------------------------------------------------
    # DIFFUSIBLE FIELDS
    # ----------------------------------------------------------

    def add_field(self, name: str, diffusion: float = 0.1, decay: float = 0.01,
                  initial_value: float = 0.0, boundary: str = "neumann"):
        """Add a diffusible chemical field."""
        f = DiffusibleField(
            name=name,
            diffusion_coefficient=diffusion,
            decay_rate=decay,
            boundary_condition=boundary
        )
        f.initialize(self.width, self.height, initial_value)
        self.fields[name] = f

    # ----------------------------------------------------------
    # HAMILTONIAN ENERGY COMPUTATION
    # ----------------------------------------------------------

    def _get_cell_type_id(self, cell_id: int) -> int:
        """Get the type ID of a cell. 0 = medium."""
        if cell_id == 0:
            return 0
        cell = self.cells.get(cell_id)
        if cell is None:
            return 0
        return cell.cell_type.type_id

    def _compute_adhesion_energy(self, x: int, y: int, cell_id: int) -> float:
        """Compute adhesion energy contribution for pixel (x,y) assigned to cell_id."""
        energy = 0.0
        type_a = self._get_cell_type_id(cell_id)

        for dx, dy in self._neighbor_offsets:
            nx = (x + dx) % self.width
            ny = (y + dy) % self.height
            neighbor_id = self.lattice[nx, ny]
            if neighbor_id != cell_id:
                type_b = self._get_cell_type_id(neighbor_id)
                energy += self.get_adhesion(type_a, type_b)

        return energy

    def _compute_volume_energy(self, cell: Cell, delta_volume: int) -> float:
        """Compute change in volume constraint energy if cell volume changes by delta."""
        ct = cell.cell_type
        v_current = cell.volume
        v_new = v_current + delta_volume
        E_old = ct.lambda_volume * (v_current - ct.target_volume) ** 2
        E_new = ct.lambda_volume * (v_new - ct.target_volume) ** 2
        return E_new - E_old

    def _compute_surface_energy_change(self, x: int, y: int,
                                        source_id: int, target_id: int) -> float:
        """Estimate surface area energy change from a pixel copy."""
        # Simplified: count boundary pixels change
        source_cell = self.cells.get(source_id)
        target_cell = self.cells.get(target_id)

        delta = 0.0

        # Estimate surface change: each neighbor that differs from the copying cell
        # changes the boundary count
        for dx, dy in self._neighbor_offsets:
            nx = (x + dx) % self.width
            ny = (y + dy) % self.height
            neighbor_id = self.lattice[nx, ny]

            if source_cell and source_id != 0:
                # Source cell loses a pixel: if neighbor was same cell, surface increases
                if neighbor_id == source_id:
                    delta += source_cell.cell_type.lambda_surface * 2
                else:
                    delta -= source_cell.cell_type.lambda_surface * 2

            if target_cell and target_id != 0:
                # Target cell gains a pixel
                if neighbor_id == target_id:
                    delta -= target_cell.cell_type.lambda_surface * 2
                else:
                    delta += target_cell.cell_type.lambda_surface * 2

        return delta * 0.1  # Scale factor for stability

    def _compute_chemotaxis_energy(self, x: int, y: int, cell_id: int) -> float:
        """Compute chemotaxis contribution: -λ_chem * Δc."""
        if cell_id == 0:
            return 0.0

        cell = self.cells.get(cell_id)
        if cell is None or cell.cell_type.chemotaxis_lambda == 0:
            return 0.0

        field_name = cell.cell_type.chemotaxis_field
        if field_name not in self.fields:
            return 0.0

        field_grid = self.fields[field_name].grid
        if field_grid is None:
            return 0.0

        lam = cell.cell_type.chemotaxis_lambda

        # Concentration at the target pixel
        c_target = field_grid[x, y]

        # Average concentration at the cell's current center
        cx = int(cell.center_x) % self.width
        cy = int(cell.center_y) % self.height
        c_source = field_grid[cx, cy]

        return -lam * (c_target - c_source)

    def _compute_delta_H(self, x: int, y: int, source_id: int, target_id: int) -> float:
        """
        Compute the total Hamiltonian change ΔH for copying pixel (x,y)
        from source_id to target_id.
        """
        delta_H = 0.0

        # 1. Adhesion energy change
        E_adhesion_old = self._compute_adhesion_energy(x, y, source_id)
        E_adhesion_new = self._compute_adhesion_energy(x, y, target_id)
        delta_H += (E_adhesion_new - E_adhesion_old)

        # 2. Volume constraint
        if source_id != 0 and source_id in self.cells:
            delta_H += self._compute_volume_energy(self.cells[source_id], -1)
        if target_id != 0 and target_id in self.cells:
            delta_H += self._compute_volume_energy(self.cells[target_id], +1)

        # 3. Surface energy (approximate)
        delta_H += self._compute_surface_energy_change(x, y, source_id, target_id)

        # 4. Chemotaxis
        delta_H += self._compute_chemotaxis_energy(x, y, target_id)

        return delta_H

    # ----------------------------------------------------------
    # MONTE CARLO METROPOLIS STEP
    # ----------------------------------------------------------

    def _attempt_pixel_copy(self):
        """
        Single Monte Carlo pixel copy attempt:
        1. Pick a random lattice site
        2. Pick a random neighbor with a different cell ID
        3. Compute ΔH for copying neighbor's ID to this site
        4. Accept with Boltzmann probability: P = min(1, exp(-ΔH/T))
        """
        x = random.randint(0, self.width - 1)
        y = random.randint(0, self.height - 1)
        source_id = self.lattice[x, y]

        # Check source cell isn't frozen
        if source_id != 0 and source_id in self.cells:
            if self.cells[source_id].cell_type.freeze:
                return

        # Pick a random neighbor
        dx, dy = random.choice(self._neighbor_offsets)
        nx = (x + dx) % self.width
        ny = (y + dy) % self.height
        target_id = self.lattice[nx, ny]

        # Only attempt if different cell IDs
        if source_id == target_id:
            return

        # Don't copy if target cell is frozen
        if target_id != 0 and target_id in self.cells:
            if self.cells[target_id].cell_type.freeze:
                return

        # Compute energy change
        delta_H = self._compute_delta_H(x, y, source_id, target_id)

        # Metropolis acceptance
        if delta_H <= 0:
            accept = True
        else:
            boltzmann_prob = math.exp(-delta_H / max(self.temperature, 0.01))
            accept = random.random() < boltzmann_prob

        if accept:
            # Perform pixel copy
            self.lattice[x, y] = target_id

            # Update cell volumes
            if source_id != 0 and source_id in self.cells:
                self.cells[source_id].volume -= 1
            if target_id != 0 and target_id in self.cells:
                self.cells[target_id].volume += 1

    def run_mcs(self, pixel_attempts_per_mcs: Optional[int] = None):
        """
        Run one Monte Carlo Step (MCS).
        By convention, one MCS = width * height pixel copy attempts.
        """
        if pixel_attempts_per_mcs is None:
            pixel_attempts_per_mcs = self.width * self.height

        for _ in range(pixel_attempts_per_mcs):
            self._attempt_pixel_copy()

        # Update cell statistics
        self._update_all_cells()

        # Cell behaviors (growth, division, death)
        self._execute_cell_behaviors()

        # Field updates (diffusion, secretion, uptake)
        self._update_fields()

        self.mcs += 1

    # ----------------------------------------------------------
    # CELL STATISTICS
    # ----------------------------------------------------------

    def _update_cell_stats(self, cell: Cell):
        """Recompute volume and center of mass for a single cell."""
        mask = self.lattice == cell.cell_id
        cell.volume = int(np.sum(mask))
        if cell.volume > 0:
            coords = np.argwhere(mask)
            cell.center_x = float(np.mean(coords[:, 0]))
            cell.center_y = float(np.mean(coords[:, 1]))
        cell.age += 1

    def _update_all_cells(self):
        """Recompute stats for all living cells."""
        # Count volumes efficiently using bincount
        flat = self.lattice.flatten()
        counts = np.bincount(flat, minlength=self.next_cell_id)

        for cell_id, cell in list(self.cells.items()):
            if not cell.alive:
                continue
            cell.volume = int(counts[cell_id]) if cell_id < len(counts) else 0

            if cell.volume == 0:
                # Cell has been squeezed out
                cell.alive = False
                continue

            # Update center of mass
            mask = self.lattice == cell_id
            coords = np.argwhere(mask)
            if len(coords) > 0:
                cell.center_x = float(np.mean(coords[:, 0]))
                cell.center_y = float(np.mean(coords[:, 1]))
            cell.age += 1

    # ----------------------------------------------------------
    # CELL BEHAVIORS
    # ----------------------------------------------------------

    def _execute_cell_behaviors(self):
        """Execute biological behaviors: growth, division, death, secretion."""
        cells_to_remove = []
        cells_to_divide = []

        for cell_id, cell in list(self.cells.items()):
            if not cell.alive:
                continue
            ct = cell.cell_type

            # Growth
            if ct.growth_rate > 0:
                cell.cell_type = CellType(
                    type_id=ct.type_id, name=ct.name,
                    target_volume=ct.target_volume + ct.growth_rate,
                    lambda_volume=ct.lambda_volume,
                    target_surface=ct.target_surface,
                    lambda_surface=ct.lambda_surface,
                    max_volume_before_division=ct.max_volume_before_division,
                    growth_rate=ct.growth_rate,
                    death_probability=ct.death_probability,
                    chemotaxis_lambda=ct.chemotaxis_lambda,
                    chemotaxis_field=ct.chemotaxis_field,
                    secretion_rates=ct.secretion_rates,
                    uptake_rates=ct.uptake_rates,
                    color=ct.color,
                    freeze=ct.freeze
                )

            # Division check
            if cell.volume >= ct.max_volume_before_division:
                cells_to_divide.append(cell_id)

            # Death check
            if ct.death_probability > 0 and random.random() < ct.death_probability:
                cells_to_remove.append(cell_id)

        # Execute divisions
        for cell_id in cells_to_divide:
            self._divide_cell(cell_id)

        # Execute deaths
        for cell_id in cells_to_remove:
            self._kill_cell(cell_id)

    def _divide_cell(self, cell_id: int):
        """Divide a cell into two daughter cells along a random axis."""
        parent = self.cells.get(cell_id)
        if parent is None or not parent.alive:
            return

        # Create daughter cell
        daughter = self.create_cell(parent.cell_type.type_id)

        # Find all pixels of the parent
        mask = self.lattice == cell_id
        coords = np.argwhere(mask)
        if len(coords) < 4:
            return

        # Split along random axis through center of mass
        cx, cy = parent.center_x, parent.center_y
        angle = random.uniform(0, math.pi)
        normal_x = math.cos(angle)
        normal_y = math.sin(angle)

        # Assign pixels to daughter based on which side of the line they're on
        for coord in coords:
            dx = coord[0] - cx
            dy = coord[1] - cy
            if dx * normal_x + dy * normal_y > 0:
                self.lattice[coord[0], coord[1]] = daughter.cell_id

        # Reset target volumes
        base_type = self.cell_types[parent.cell_type.type_id]
        parent.cell_type = CellType(
            type_id=base_type.type_id, name=base_type.name,
            target_volume=base_type.target_volume,
            lambda_volume=base_type.lambda_volume,
            target_surface=base_type.target_surface,
            lambda_surface=base_type.lambda_surface,
            max_volume_before_division=base_type.max_volume_before_division,
            growth_rate=base_type.growth_rate,
            death_probability=base_type.death_probability,
            chemotaxis_lambda=base_type.chemotaxis_lambda,
            chemotaxis_field=base_type.chemotaxis_field,
            secretion_rates=base_type.secretion_rates,
            uptake_rates=base_type.uptake_rates,
            color=base_type.color
        )

        self._update_cell_stats(parent)
        self._update_cell_stats(daughter)

    def _kill_cell(self, cell_id: int):
        """Remove a cell from the lattice (apoptosis)."""
        cell = self.cells.get(cell_id)
        if cell is None:
            return
        cell.alive = False
        # Clear pixels
        self.lattice[self.lattice == cell_id] = 0

    # ----------------------------------------------------------
    # FIELD UPDATES (DIFFUSION + CELL COUPLING)
    # ----------------------------------------------------------

    def _update_fields(self):
        """Update all diffusible fields: diffusion, decay, secretion, uptake."""
        for field_name, field_obj in self.fields.items():
            # Cell secretion and uptake
            for cell_id, cell in self.cells.items():
                if not cell.alive:
                    continue
                ct = cell.cell_type

                # Secretion: add to field at cell's pixels
                if field_name in ct.secretion_rates:
                    rate = ct.secretion_rates[field_name]
                    mask = self.lattice == cell_id
                    field_obj.grid[mask] += rate

                # Uptake: remove from field at cell's pixels
                if field_name in ct.uptake_rates:
                    rate = ct.uptake_rates[field_name]
                    mask = self.lattice == cell_id
                    field_obj.grid[mask] = np.maximum(
                        0, field_obj.grid[mask] - rate
                    )

            # Diffusion + decay step
            field_obj.diffuse_and_decay()

    # ----------------------------------------------------------
    # SIMULATION RUNNER
    # ----------------------------------------------------------

    def simulate(
        self,
        num_mcs: int = 100,
        save_every: int = 10,
        pixel_attempts_factor: float = 1.0
    ) -> Dict[str, Any]:
        """
        Run the full CPM simulation.

        Args:
            num_mcs: Number of Monte Carlo steps
            save_every: Save state every N MCS
            pixel_attempts_factor: Multiply default pixel attempts (for speed tuning)

        Returns:
            Dict with lattice history, field history, cell counts, etc.
        """
        attempts_per_mcs = int(self.width * self.height * pixel_attempts_factor)

        lattice_history = []
        field_history = {name: [] for name in self.fields}
        cell_count_history = []
        time_points = []

        # Save initial state
        lattice_history.append(self._get_lattice_colored())
        for name, f in self.fields.items():
            field_history[name].append(f.grid.copy().tolist() if f.grid is not None else [])
        cell_count_history.append(self._get_cell_counts())
        time_points.append(0)

        for mcs_step in range(1, num_mcs + 1):
            self.run_mcs(attempts_per_mcs)

            if mcs_step % save_every == 0 or mcs_step == num_mcs:
                lattice_history.append(self._get_lattice_colored())
                for name, f in self.fields.items():
                    field_history[name].append(
                        f.grid.copy().tolist() if f.grid is not None else []
                    )
                cell_count_history.append(self._get_cell_counts())
                time_points.append(mcs_step)

        return {
            "type": "ABM",
            "width": self.width,
            "height": self.height,
            "t": time_points,
            "lattice_frames": lattice_history,
            "fields": field_history,
            "cell_counts": cell_count_history,
            "cell_types": {
                ct.type_id: {
                    "name": ct.name,
                    "color": list(ct.color)
                } for ct in self.cell_types.values()
            },
            "total_mcs": num_mcs,
            "temperature": self.temperature
        }

    def _get_lattice_colored(self) -> List[List[List[int]]]:
        """Convert lattice to RGB color array for visualization."""
        colored = np.zeros((self.width, self.height, 3), dtype=np.uint8)

        # Medium color
        medium_color = self.cell_types[0].color
        colored[:, :] = medium_color

        for cell_id, cell in self.cells.items():
            if not cell.alive:
                continue
            mask = self.lattice == cell_id
            colored[mask] = cell.cell_type.color

        return colored.tolist()

    def _get_cell_counts(self) -> Dict[str, int]:
        """Count cells by type."""
        counts = {}
        for ct_id, ct in self.cell_types.items():
            if ct_id == 0:
                continue
            counts[ct.name] = sum(
                1 for c in self.cells.values()
                if c.alive and c.cell_type.type_id == ct_id
            )
        counts["total"] = sum(1 for c in self.cells.values() if c.alive)
        return counts


# ============================================================
# BLUEPRINT TO CPM CONVERTER
# ============================================================

def build_cpm_from_blueprint(blueprint: Dict[str, Any]) -> CellularPottsModel:
    """
    Convert an ABM blueprint dict into a configured CellularPottsModel.

    Blueprint schema:
    {
        "type": "ABM",
        "grid": {"width": 100, "height": 100},
        "temperature": 10.0,
        "cell_types": [
            {
                "type_id": 1,
                "name": "Cancer",
                "target_volume": 25,
                "lambda_volume": 2.0,
                "growth_rate": 0.1,
                "death_probability": 0.001,
                "color": [255, 50, 50],
                "secretion": {"nutrient": -0.01},
                "chemotaxis": {"field": "nutrient", "lambda": 100}
            }
        ],
        "adhesion_matrix": {
            "0-1": 16, "1-1": 2, "1-2": 11, ...
        },
        "initial_config": [
            {"type_id": 1, "count": 20, "radius": 3, "region": "center"},
            {"type_id": 2, "count": 50, "radius": 3, "region": "random"}
        ],
        "fields": [
            {"name": "nutrient", "diffusion": 0.2, "decay": 0.01, "initial": 1.0}
        ],
        "simulation_config": {
            "num_mcs": 500,
            "save_every": 10
        }
    }
    """
    grid = blueprint.get("grid", {})
    width = grid.get("width", 100)
    height = grid.get("height", 100)
    temp = blueprint.get("temperature", 10.0)

    cpm = CellularPottsModel(width=width, height=height, temperature=temp)

    # Register cell types
    for ct_def in blueprint.get("cell_types", []):
        secretion = ct_def.get("secretion", {})
        uptake = {}
        sec_rates = {}
        for field_name, rate in secretion.items():
            if rate < 0:
                uptake[field_name] = abs(rate)
            else:
                sec_rates[field_name] = rate

        chemo = ct_def.get("chemotaxis", {})

        ct = CellType(
            type_id=ct_def["type_id"],
            name=ct_def.get("name", f"Type_{ct_def['type_id']}"),
            target_volume=ct_def.get("target_volume", 25),
            lambda_volume=ct_def.get("lambda_volume", 2.0),
            target_surface=ct_def.get("target_surface", 20),
            lambda_surface=ct_def.get("lambda_surface", 1.0),
            max_volume_before_division=ct_def.get("max_volume_before_division", 50),
            growth_rate=ct_def.get("growth_rate", 0.0),
            death_probability=ct_def.get("death_probability", 0.0),
            chemotaxis_lambda=chemo.get("lambda", 0.0),
            chemotaxis_field=chemo.get("field", ""),
            secretion_rates=sec_rates,
            uptake_rates=uptake,
            color=tuple(ct_def.get("color", [100, 100, 255]))
        )
        cpm.register_cell_type(ct)

    # Adhesion matrix
    for key, energy in blueprint.get("adhesion_matrix", {}).items():
        parts = key.split("-")
        if len(parts) == 2:
            cpm.set_adhesion(int(parts[0]), int(parts[1]), energy)

    # Fields
    for f_def in blueprint.get("fields", []):
        cpm.add_field(
            name=f_def["name"],
            diffusion=f_def.get("diffusion", 0.1),
            decay=f_def.get("decay", 0.01),
            initial_value=f_def.get("initial", 0.0),
            boundary=f_def.get("boundary", "neumann")
        )

    # Initial cell placement
    for init in blueprint.get("initial_config", []):
        type_id = init["type_id"]
        count = init.get("count", 10)
        radius = init.get("radius", 3)
        region = init.get("region", "random")

        if region == "center":
            # Place cells near center
            cx, cy = width // 2, height // 2
            spread = int(math.sqrt(count) * radius * 2)
            for i in range(count):
                cell = cpm.create_cell(type_id)
                px = cx + random.randint(-spread, spread)
                py = cy + random.randint(-spread, spread)
                px = max(radius, min(width - radius - 1, px))
                py = max(radius, min(height - radius - 1, py))
                cpm.seed_cell_at(cell, px, py, radius)
        elif region == "random":
            cpm.seed_random_cells(type_id, count, radius)
        elif region == "ring":
            # Place cells in a ring
            ring_r = min(width, height) // 3
            for i in range(count):
                angle = 2 * math.pi * i / count
                px = int(width / 2 + ring_r * math.cos(angle))
                py = int(height / 2 + ring_r * math.sin(angle))
                px = max(radius, min(width - radius - 1, px))
                py = max(radius, min(height - radius - 1, py))
                cell = cpm.create_cell(type_id)
                cpm.seed_cell_at(cell, px, py, radius)
        elif region == "block":
            x0 = init.get("x0", 0)
            y0 = init.get("y0", 0)
            x1 = init.get("x1", width // 2)
            y1 = init.get("y1", height // 2)
            cpm.seed_block(type_id, x0, y0, x1, y1, cell_size=radius * 2)

    return cpm
