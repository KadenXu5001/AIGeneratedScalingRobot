import numpy as np
import copy
import os
from argparse import ArgumentParser
from heightSim import HeightSimulator
from utils import load_config
from robot import load_robots

def mutate_robot(robot_data, mutation_rate=0.2, jitter_amount=0.08):
    new_robot = copy.deepcopy(robot_data)
    masses = np.array(new_robot["masses"])
    mask = np.random.rand(*masses.shape) < mutation_rate
    jitter = np.random.uniform(-jitter_amount, jitter_amount, masses.shape)
    masses += jitter * mask
    masses[:, 1] = np.maximum(masses[:, 1], 0.05) 
    new_robot["masses"] = masses.tolist()
    new_robot["n_masses"] = len(new_robot["masses"])
    return new_robot

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--generations", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(config["seed"])
    pop_size = config["simulator"]["n_sims"]
    
    # Generation 0: Initial random or loaded robots
    current_robots = load_robots(num_robots=pop_size)

    for gen in range(args.generations):
        print(f"\n{'='*20} GENERATION {gen} {'='*20}")

        # 1. Setup metadata for GPU allocation
        num_masses = [r["n_masses"] for r in current_robots]
        num_springs = [r["n_springs"] for r in current_robots]
        max_num_masses, max_num_springs = max(num_masses), max(num_springs)
        config["simulator"]["n_masses"] = max_num_masses
        config["simulator"]["n_springs"] = max_num_springs

        # 2. Re-initialize Simulator
        simulator = HeightSimulator(sim_config=config["simulator"], taichi_config=config["taichi"], seed=config["seed"])
        simulator.initialize([r["masses"] for r in current_robots], [r["springs"] for r in current_robots])

        # 3. Handle Brain Inheritance (Lamarckian)
        for i, robot in enumerate(current_robots):
            if "control_params" in robot:
                simulator.set_control_params([i], [robot["control_params"]])

        # --- CAPTURE "BEFORE" STATE FOR ALL ---
        # We grab these now so we can save the top 5's original state later
        initial_params_all = simulator.get_control_params(range(pop_size))

        # 4. Train
        print(f"Training Generation {gen}...")
        fitness_history = simulator.train() 
        final_fitness = fitness_history[:, -1]

        # 5. Selection
        top_5_indices = np.argsort(final_fitness)[-5:][::-1]
        
        # 6. Capture "AFTER" State for Top 5
        final_params_top_5 = simulator.get_control_params(top_5_indices)

        winners = []
        for i, rank_idx in enumerate(top_5_indices):
            # Base robot data (Morphology)
            robot_meta = current_robots[rank_idx].copy()
            robot_meta["max_n_masses"] = max_num_masses
            robot_meta["max_n_springs"] = max_num_springs

            # A. Save "Before" version (Morphology + Initial Weights)
            meta_before = robot_meta.copy()
            meta_before["control_params"] = initial_params_all[rank_idx]
            np.save(f"gen{gen}_rank{i+1}_before.npy", meta_before)

            # B. Save "After" version (Morphology + Trained Weights)
            meta_after = robot_meta.copy()
            meta_after["control_params"] = final_params_top_5[i]
            np.save(f"gen{gen}_rank{i+1}_after.npy", meta_after)
            
            # Store for reproduction
            winners.append(meta_after)

            if i == 0:
                print(f"Gen {gen} Top Fitness: {final_fitness[rank_idx]:.4f}")

        # 7. Reproduction
        next_gen = []
        next_gen.extend(winners) # Elitism
        while len(next_gen) < pop_size:
            parent = np.random.choice(winners)
            next_gen.append(mutate_robot(parent))
        
        current_robots = next_gen
        del simulator # Clean up GPU memory

    print("\nFull Evolution and Save cycle complete.")