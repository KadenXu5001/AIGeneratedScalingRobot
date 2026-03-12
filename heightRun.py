from heightSim import HeightSimulator
from utils import load_config
from argparse import ArgumentParser
from robot import load_robots
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Set the random seed for reproducibility
    np.random.seed(config["seed"])
    robots = load_robots(num_robots=config["simulator"]["n_sims"])

    # Extract the number of masses and springs
    num_masses = [robot["n_masses"] for robot in robots]
    num_springs = [robot["n_springs"] for robot in robots]
    max_num_masses = max(num_masses)
    max_num_springs = max(num_springs)
    
    config["simulator"]["n_masses"] = max_num_masses
    config["simulator"]["n_springs"] = max_num_springs

    # Initialize the simulator
    simulator = HeightSimulator(sim_config=config["simulator"], taichi_config=config["taichi"], seed=config["seed"], needs_grad=True)

    masses = [robot["masses"] for robot in robots]
    springs = [robot["springs"] for robot in robots]
    simulator.initialize(masses, springs)

    # --- CAPTURE "BEFORE" STATE ---
    # We get the control parameters right after initialization
    initial_control_params = simulator.get_control_params(range(len(robots)))

    print(f"springK={config['simulator']['springK']}, lr={config['simulator']['learning_rate']}, drag={config['simulator']['drag_damping']}")

    # Train the robots
    fitness_history = simulator.train() 
    np.save("fitness_history.npy", fitness_history)

    # --- IDENTIFY BEST ROBOT ---
    fitness = fitness_history[:, -1]
    best_idx = np.argmax(fitness) # Index of the #1 performer
    best_robot_meta = robots[best_idx]
    
    # --- CAPTURE "AFTER" STATE ---
    

# 1. Get the indices of the top 5 robots based on fitness (descending)
    top_5_indices = np.argsort(fitness)[-5:][::-1]

    # This returns a list of length 5
    final_control_params = simulator.get_control_params(top_5_indices)

    for i, rank_idx in enumerate(top_5_indices):
        # 1. Setup Base Metadata (use the specific robot's metadata, not best_robot_meta)
        robot_meta = robots[rank_idx].copy() 
        robot_meta["max_n_masses"] = max_num_masses
        robot_meta["max_n_springs"] = max_num_springs
        
        # 2. Save "BEFORE" version
        # Use [rank_idx] because initial_control_params contains the WHOLE population
        meta_before = robot_meta.copy()
        meta_before["control_params"] = initial_control_params[rank_idx]
        np.save(f"top_{i+1}_robot_before.npy", meta_before)
        
        # 3. Save "AFTER" version
        # Use [i] because final_control_params ONLY contains the 5 you just requested
        meta_after = robot_meta.copy()
        meta_after["control_params"] = final_control_params[i] 
        np.save(f"top_{i+1}_robot_after.npy", meta_after)

        print(f"Saved Rank {i+1} (Index {rank_idx}): Fitness {fitness[rank_idx]}")

    print(f"Saved best robot (Index {best_idx}) states. Final Fitness: {fitness[best_idx]}")