import numpy as np
import copy

def mutate_robot(robot_data, mutation_rate=0.1, jitter_amount=0.05):
    """
    Creates a slightly different version of a robot's body.
    """
    new_robot = copy.deepcopy(robot_data)
    masses = np.array(new_robot["masses"])
    
    # 1. Jitter Mass Positions (Morphological Mutation)
    # Only move masses slightly so we don't destroy the structure
    mask = np.random.rand(*masses.shape) < mutation_rate
    jitter = np.random.uniform(-jitter_amount, jitter_amount, masses.shape)
    masses += jitter * mask
    
    # Keep the robot above ground and centered
    masses[:, 1] = np.maximum(masses[:, 1], 0.05) 
    new_robot["masses"] = masses.tolist()
    
    # 2. Reset Brain for the new body? 
    # Usually, we keep the old brain as a starting point, 
    # but the simulator will re-train it anyway.
    return new_robot

def generate_next_generation(top_robots_paths, population_size):
    new_population = []
    
    # Load the winners
    winners = [np.load(p, allow_pickle=True).item() for p in top_robots_paths]
    
    # Fill the new population by mutating the winners
    while len(new_population) < population_size:
        parent = np.random.choice(winners)
        child = mutate_robot(parent)
        new_population.append(child)
        
    return new_population