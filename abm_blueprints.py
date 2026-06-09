"""
Predefined ABM Blueprint Templates for the Cellular Potts Model.

Provides ready-to-run demonstrations of:
1. Cell Sorting (differential adhesion hypothesis)
2. Tumor Growth with nutrient-dependent dynamics
3. Wound Healing with chemotactic migration
4. PDAC Tumor Microenvironment (cancer + fibroblasts + immune cells)
"""

from typing import Dict, Any, List


def get_cell_sorting_blueprint() -> Dict[str, Any]:
    """
    Classic differential adhesion cell sorting.
    Two cell types with different adhesion energies spontaneously sort
    into distinct clusters (Steinberg's differential adhesion hypothesis).
    """
    return {
        "type": "ABM",
        "name": "Cell Sorting (Differential Adhesion)",
        "description": "Two cell types with different adhesion energies sort into clusters. "
                       "Light cells (Type 1) are more cohesive than dark cells (Type 2).",
        "grid": {"width": 80, "height": 80},
        "temperature": 15.0,
        "cell_types": [
            {
                "type_id": 1,
                "name": "LightCells",
                "target_volume": 25,
                "lambda_volume": 2.5,
                "target_surface": 20,
                "lambda_surface": 0.5,
                "max_volume_before_division": 200,  # Don't divide
                "growth_rate": 0.0,
                "death_probability": 0.0,
                "color": [0, 200, 255]
            },
            {
                "type_id": 2,
                "name": "DarkCells",
                "target_volume": 25,
                "lambda_volume": 2.5,
                "target_surface": 20,
                "lambda_surface": 0.5,
                "max_volume_before_division": 200,
                "growth_rate": 0.0,
                "death_probability": 0.0,
                "color": [200, 50, 100]
            }
        ],
        "adhesion_matrix": {
            "0-0": 0,    # Medium-Medium
            "0-1": 16,   # Medium-LightCells
            "0-2": 16,   # Medium-DarkCells
            "1-1": 2,    # LightCells-LightCells (strong adhesion)
            "1-2": 11,   # LightCells-DarkCells (weak)
            "2-2": 14    # DarkCells-DarkCells (moderate adhesion)
        },
        "initial_config": [
            {"type_id": 1, "count": 30, "radius": 3, "region": "random"},
            {"type_id": 2, "count": 30, "radius": 3, "region": "random"}
        ],
        "fields": [],
        "simulation_config": {
            "num_mcs": 500,
            "save_every": 10
        }
    }


def get_tumor_growth_blueprint() -> Dict[str, Any]:
    """
    Tumor growth with nutrient-dependent proliferation.
    Cancer cells consume nutrient, grow, and divide. Cells in nutrient-poor
    regions die, forming a necrotic core surrounded by a proliferative rim.
    """
    return {
        "type": "ABM",
        "name": "Tumor Growth (Nutrient-Dependent)",
        "description": "Cancer cells grow and divide when nutrient is available. "
                       "Cells in nutrient-poor regions die, forming a necrotic core.",
        "grid": {"width": 100, "height": 100},
        "temperature": 12.0,
        "cell_types": [
            {
                "type_id": 1,
                "name": "Cancer",
                "target_volume": 25,
                "lambda_volume": 3.0,
                "target_surface": 22,
                "lambda_surface": 1.0,
                "max_volume_before_division": 50,
                "growth_rate": 0.3,
                "death_probability": 0.002,
                "color": [230, 60, 60],
                "secretion": {"nutrient": -0.02},  # Negative = uptake
                "chemotaxis": {"field": "nutrient", "lambda": 200}
            }
        ],
        "adhesion_matrix": {
            "0-0": 0,
            "0-1": 20,
            "1-1": 5
        },
        "initial_config": [
            {"type_id": 1, "count": 15, "radius": 3, "region": "center"}
        ],
        "fields": [
            {
                "name": "nutrient",
                "diffusion": 0.2,
                "decay": 0.001,
                "initial": 1.0,
                "boundary": "neumann"
            }
        ],
        "simulation_config": {
            "num_mcs": 800,
            "save_every": 20
        }
    }


def get_wound_healing_blueprint() -> Dict[str, Any]:
    """
    Wound healing model with chemotactic migration.
    Epithelial cells migrate into a wound gap following a chemokine gradient
    produced by the wound edge.
    """
    return {
        "type": "ABM",
        "name": "Wound Healing (Chemotaxis)",
        "description": "Epithelial cells migrate into a wound gap following "
                       "a chemokine gradient. Demonstrates collective cell migration.",
        "grid": {"width": 120, "height": 80},
        "temperature": 10.0,
        "cell_types": [
            {
                "type_id": 1,
                "name": "Epithelial",
                "target_volume": 30,
                "lambda_volume": 2.0,
                "target_surface": 24,
                "lambda_surface": 1.0,
                "max_volume_before_division": 55,
                "growth_rate": 0.1,
                "death_probability": 0.0,
                "color": [80, 200, 120],
                "chemotaxis": {"field": "chemokine", "lambda": 300}
            }
        ],
        "adhesion_matrix": {
            "0-0": 0,
            "0-1": 25,
            "1-1": 6
        },
        "initial_config": [
            # Left block of cells
            {"type_id": 1, "count": 0, "radius": 3, "region": "block",
             "x0": 0, "y0": 0, "x1": 40, "y1": 80},
            # Right block of cells (wound gap in between)
            {"type_id": 1, "count": 0, "radius": 3, "region": "block",
             "x0": 80, "y0": 0, "x1": 120, "y1": 80}
        ],
        "fields": [
            {
                "name": "chemokine",
                "diffusion": 0.3,
                "decay": 0.005,
                "initial": 0.0,
                "boundary": "neumann"
            }
        ],
        "simulation_config": {
            "num_mcs": 600,
            "save_every": 15
        }
    }


def get_pdac_tme_blueprint() -> Dict[str, Any]:
    """
    Pancreatic Ductal Adenocarcinoma (PDAC) Tumor Microenvironment.
    Three cell types: cancer cells, fibroblasts (CAFs), and immune cells (CD8+ T cells).
    Connects to MAPLE calibration targets for the Popel lab's QSP model.
    """
    return {
        "type": "ABM",
        "name": "PDAC Tumor Microenvironment",
        "description": "Three-cell-type model of the pancreatic tumor microenvironment: "
                       "cancer cells, cancer-associated fibroblasts (CAFs), and CD8+ T cells. "
                       "Cancer cells proliferate and secrete growth factors. CAFs produce ECM "
                       "and TGF-β. T cells migrate via chemotaxis and kill cancer cells.",
        "grid": {"width": 120, "height": 120},
        "temperature": 10.0,
        "cell_types": [
            {
                "type_id": 1,
                "name": "Cancer",
                "target_volume": 30,
                "lambda_volume": 3.0,
                "target_surface": 25,
                "lambda_surface": 1.0,
                "max_volume_before_division": 55,
                "growth_rate": 0.2,
                "death_probability": 0.001,
                "color": [230, 60, 60],
                "secretion": {"growth_factor": 0.05, "nutrient": -0.01},
                "chemotaxis": {"field": "nutrient", "lambda": 50}
            },
            {
                "type_id": 2,
                "name": "Fibroblast",
                "target_volume": 20,
                "lambda_volume": 2.5,
                "target_surface": 18,
                "lambda_surface": 0.8,
                "max_volume_before_division": 200,  # Rarely divide
                "growth_rate": 0.0,
                "death_probability": 0.0005,
                "color": [80, 160, 255],
                "secretion": {"tgfb": 0.03},
                "chemotaxis": {"field": "growth_factor", "lambda": 80}
            },
            {
                "type_id": 3,
                "name": "CD8_Tcell",
                "target_volume": 15,
                "lambda_volume": 4.0,
                "target_surface": 14,
                "lambda_surface": 1.5,
                "max_volume_before_division": 200,  # Don't divide in ABM
                "growth_rate": 0.0,
                "death_probability": 0.005,  # Exhaustion/death
                "color": [0, 230, 100],
                "secretion": {},
                "chemotaxis": {"field": "growth_factor", "lambda": 250}
            }
        ],
        "adhesion_matrix": {
            "0-0": 0,
            "0-1": 18, "0-2": 18, "0-3": 20,
            "1-1": 4,   # Cancer-Cancer: strong adhesion
            "1-2": 10,  # Cancer-Fibroblast: moderate
            "1-3": 15,  # Cancer-Tcell: contact killing interface
            "2-2": 6,   # Fibroblast-Fibroblast
            "2-3": 14,  # Fibroblast-Tcell
            "3-3": 12   # Tcell-Tcell
        },
        "initial_config": [
            {"type_id": 1, "count": 20, "radius": 3, "region": "center"},
            {"type_id": 2, "count": 15, "radius": 2, "region": "ring"},
            {"type_id": 3, "count": 10, "radius": 2, "region": "random"}
        ],
        "fields": [
            {"name": "nutrient", "diffusion": 0.3, "decay": 0.005, "initial": 1.0},
            {"name": "growth_factor", "diffusion": 0.1, "decay": 0.02, "initial": 0.0},
            {"name": "tgfb", "diffusion": 0.15, "decay": 0.01, "initial": 0.0}
        ],
        "simulation_config": {
            "num_mcs": 1000,
            "save_every": 25
        }
    }


# Registry of all presets
ABM_PRESETS = {
    "cell_sorting": get_cell_sorting_blueprint,
    "tumor_growth": get_tumor_growth_blueprint,
    "wound_healing": get_wound_healing_blueprint,
    "pdac_tme": get_pdac_tme_blueprint
}


def get_abm_preset(name: str) -> Dict[str, Any]:
    """Get an ABM preset blueprint by name."""
    if name not in ABM_PRESETS:
        raise ValueError(f"Unknown ABM preset: {name}. Available: {list(ABM_PRESETS.keys())}")
    return ABM_PRESETS[name]()


def list_abm_presets() -> List:
    """List available ABM presets with metadata."""
    from typing import List as TList
    result = []
    for key, factory in ABM_PRESETS.items():
        bp = factory()
        result.append({
            "id": key,
            "name": bp.get("name", key),
            "description": bp.get("description", ""),
            "cell_types": [ct["name"] for ct in bp.get("cell_types", [])],
            "num_mcs": bp.get("simulation_config", {}).get("num_mcs", 500)
        })
    return result
