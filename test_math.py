import numpy as np
from simulation_engine import ODEModel, solve_pde
import json

def test_ode_compilation_and_simulation():
    print("Testing ODE Compilation & Simulation...")
    
    # 3-node pathway EGF -> EGFR -> ERK
    blueprint = {
        "nodes": [
            {"id": "EGF", "name": "EGF ligand", "initial_value": 10.0},
            {"id": "EGFR", "name": "EGF receptor", "initial_value": 1.0},
            {"id": "ERK", "name": "ERK kinase", "initial_value": 0.0}
        ],
        "edges": [
            {"source": "EGF", "target": "EGFR", "type": "activation", "parameters": {"k": 0.5, "K_d": 1.0, "n": 1.0}},
            {"source": "EGFR", "target": "ERK", "type": "activation", "parameters": {"k": 0.5, "K_d": 1.0, "n": 2.0}}
        ]
    }
    
    # Compile
    model = ODEModel(blueprint)
    
    # Check compiled parameter keys
    expected_params = [
        "act_EGF_to_EGFR_Kd", "act_EGF_to_EGFR_k", "act_EGF_to_EGFR_n",
        "act_EGFR_to_ERK_Kd", "act_EGFR_to_ERK_k", "act_EGFR_to_ERK_n",
        "deg_EGF", "deg_EGFR", "deg_ERK"
    ]
    for p in expected_params:
        assert p in model.param_names, f"Missing parameter {p} in compiled system!"
        
    print("[OK] ODE Symbol compilation successful.")
    
    # Simulate
    results = model.simulate(t_max=10.0, num_points=50)
    assert len(results["t"]) == 50
    assert "EGF" in results["species"]
    assert "EGFR" in results["species"]
    assert "ERK" in results["species"]
    
    # Check that EGF decreases (due to default degradation) and ERK increases (due to activation cascade)
    assert results["species"]["EGF"][-1] < results["species"]["EGF"][0]
    assert results["species"]["ERK"][-1] > results["species"]["ERK"][0]
    
    print("[OK] ODE Numerical simulation successful.")


def test_parameter_optimization():
    print("Testing Parameter Optimization...")
    
    blueprint = {
        "nodes": [
            {"id": "A", "initial_value": 5.0},
            {"id": "B", "initial_value": 0.0}
        ],
        "edges": [
            {"source": "A", "target": "B", "type": "activation", "parameters": {"k": 1.0, "K_d": 1.0, "n": 1.0}}
        ]
    }
    
    model = ODEModel(blueprint)
    
    # Generate some artificial target data with k_true = 2.0
    times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    true_params = {"act_A_to_B_k": 2.5, "deg_B": 0.2}
    sim_true = model.simulate(t_max=10.0, num_points=6, custom_params=true_params)
    
    target_data = {
        "B": sim_true["species"]["B"]
    }
    
    # Optimize starting from k = 1.0, deg_B = 0.1
    params_to_fit = ["act_A_to_B_k", "deg_B"]
    fitted, loss = model.fit_parameters_to_target(target_data, times, params_to_fit)
    
    print(f"Fitted parameters: {fitted}, Final Loss: {loss:.5f}")
    assert loss < 1e-3, "Optimization failed to minimize loss!"
    assert abs(fitted["act_A_to_B_k"] - 2.5) < 0.1, "Optimized parameter mismatch!"
    
    print("[OK] Parameter fitting optimization successful.")


def test_pde_solver():
    print("Testing 2D PDE Reaction-Diffusion Solver...")
    
    spatial = {
        "x_grid": 10,
        "y_grid": 10,
        "dx": 1.0,
        "dy": 1.0,
        "diffusion": {
            "u": 0.1,
            "v": 0.5
        }
    }
    reactions = {
        "u": "u**2 / v - u + 0.02",
        "v": "u**2 - v"
    }
    initial_conditions = {
        "u": {"type": "random_noise", "base_value": 1.0, "noise_amplitude": 0.01},
        "v": {"type": "random_noise", "base_value": 1.0, "noise_amplitude": 0.01}
    }
    
    result = solve_pde(
        spatial_config=spatial,
        reaction_formulas=reactions,
        initial_conditions=initial_conditions,
        t_max=5.0,
        dt=0.1,
        save_every=10
    )
    
    assert result["x_size"] == 10
    assert result["y_size"] == 10
    assert "u" in result["species"]
    assert "v" in result["species"]
    # Check we got the correct number of saved frames: initial (t=0) + 5 steps of 1.0s (since save_every=10, dt=0.1, saved at step 10, 20, 30, 40, 50)
    assert len(result["t"]) == 6
    
    print("[OK] 2D reaction-diffusion PDE solver successful.")


if __name__ == "__main__":
    test_ode_compilation_and_simulation()
    test_parameter_optimization()
    test_pde_solver()
    print("\nALL MATH VERIFICATION TESTS PASSED SUCCESSFULLY!")
