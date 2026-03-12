import os, numpy as np
os.environ["ENABLE_TAICHI_HEADER_PRINT"] = "False"  # Suppress Taichi startup header output
import taichi as ti
from tqdm import tqdm  # Progress bar library for training loop visualization

# Map string architecture names to Taichi backend constants
architectures = {
    "cuda": ti.cuda,   # NVIDIA GPU backend
    "cpu": ti.cpu,     # CPU backend
    "metal": ti.metal  # Apple Metal GPU backend
}

# Shorthand type alias: 2D float32 vector used for positions and velocities
vec2 = ti.types.vector(2, ti.f32)

TEMPERATURE = 50.0  # log-sum-exp temperature: higher = closer to true max height, lower = smoother gradients

@ti.data_oriented  # Marks class as containing Taichi kernels and fields
class HeightSimulator:
    def __init__(self, sim_config, taichi_config, seed, needs_grad=True):
        # Initialize Taichi with chosen backend, float32 precision, and fixed random seed
        ti.init(
            arch=architectures[taichi_config["arch"]],
            default_fp=ti.f32,
            random_seed=seed,
            **taichi_config["init"],  # Pass any extra Taichi init options from config
        )
        self.needs_grad = needs_grad      # Whether to allocate gradient fields (needed for training)
        self.config = sim_config          # Store simulation hyperparameter config dict
        self.taichi_config = taichi_config # Store Taichi backend config dict
        self.set_constants()              # Create and fill scalar Taichi fields from config
        self.allocate_fields()            # Allocate all simulation state arrays

    def set_constants(self):
        # --- Scalar Taichi fields for simulation constants ---
        # These are stored as Taichi fields (not plain Python vars) so GPU kernels can read them

        self.n_sims = ti.field(dtype=ti.i32, shape=(), needs_grad=False)         # Number of parallel simulations
        self.steps = ti.field(dtype=ti.i32, shape=(), needs_grad=False)           # Number of timesteps per sim
        self.max_n_masses = ti.field(dtype=ti.i32, shape=(), needs_grad=False)    # Max masses per robot (for array sizing)
        self.max_n_springs = ti.field(dtype=ti.i32, shape=(), needs_grad=False)   # Max springs per robot
        self.ground_height = ti.field(dtype=ti.f32, shape=(), needs_grad=False)   # Y-coordinate of the ground plane
        self.dt = ti.field(dtype=ti.f32, shape=(), needs_grad=False)              # Timestep size in seconds
        self.springA = ti.field(dtype=ti.f32, shape=(), needs_grad=False)         # Max spring actuation amplitude
        self.springK = ti.field(dtype=ti.f32, shape=(), needs_grad=False)         # Spring stiffness constant
        self.gravity = ti.field(dtype=ti.f32, shape=(), needs_grad=False)         # Gravitational acceleration magnitude
        self.friction = ti.field(dtype=ti.f32, shape=(), needs_grad=False)        # Friction coefficient for ground contact
        self.restitution = ti.field(dtype=ti.f32, shape=(), needs_grad=False)     # Bounciness coefficient (0=no bounce, 1=elastic)
        self.drag_damping = ti.field(dtype=ti.f32, shape=(), needs_grad=False)    # Air drag damping coefficient
        self.eps = ti.field(dtype=ti.f32, shape=(), needs_grad=False)             # Small epsilon to prevent divide-by-zero
        self.nn_hidden_size = ti.field(dtype=ti.i32, shape=(), needs_grad=False)  # Max hidden layer size for neural net
        self.nn_cpg_count = ti.field(dtype=ti.i32, shape=(), needs_grad=False)    # Number of CPG oscillator inputs
        self.cpg_omega = ti.field(dtype=ti.f32, shape=(), needs_grad=False)       # Angular frequency of CPG oscillators
        self.adam_beta1 = ti.field(dtype=ti.f32, shape=(), needs_grad=False)      # Adam optimizer: exponential decay for 1st moment
        self.adam_beta2 = ti.field(dtype=ti.f32, shape=(), needs_grad=False)      # Adam optimizer: exponential decay for 2nd moment
        self.learning_rate = ti.field(dtype=ti.f32, shape=(), needs_grad=False)   # Learning rate for weight updates
        self.rung_elev = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        self.rung_half_distance = ti.field(dtype=ti.f32, shape=(), needs_grad=False)
        # --- Copy values from Python config dict into Taichi fields ---
        self.n_sims[None] = self.config["n_sims"]
        self.steps[None] = self.config["sim_steps"]
        self.max_n_masses[None] = self.config["n_masses"]
        self.max_n_springs[None] = self.config["n_springs"]
        self.ground_height[None] = self.config["ground_height"]
        self.dt[None] = self.config["dt"]
        self.springA[None] = self.config["springA"]
        self.springK[None] = self.config["springK"]
        self.gravity[None] = self.config["gravity"]
        self.friction[None] = self.config["friction"]
        self.restitution[None] = self.config["restitution"]
        self.drag_damping[None] = self.config["drag_damping"]
        self.eps[None] = self.config["eps"]
        self.nn_hidden_size[None] = self.config["nn_hidden_size"]
        self.nn_cpg_count[None] = self.config["nn_cpg_count"]
        self.cpg_omega[None] = self.config["cpg_omega"]
        self.adam_beta1[None] = self.config["adam_beta1"]
        self.adam_beta2[None] = self.config["adam_beta2"]
        self.learning_rate[None] = self.config["learning_rate"]
        self.rung_elev[None] = self.config.get("rung_elevation", 0.2)
        self.rung_half_distance[None] = self.config.get("rung_half_distance", 0.25)

    def allocate_fields(self):
        # --- Structural fields (no gradients needed) ---
        self.springs = ti.Vector.field(2, dtype=ti.i32, shape=(self.n_sims[None], self.max_n_springs[None],), needs_grad=False)   # Spring endpoint mass indices [mass_a, mass_b]
        self.springL = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.max_n_springs[None],), needs_grad=False)             # Rest length of each spring
        self.n_masses = ti.field(dtype=ti.i32, shape=(self.n_sims[None],), needs_grad=False)                                      # Actual number of masses per robot
        self.n_springs = ti.field(dtype=ti.i32, shape=(self.n_sims[None],), needs_grad=False)                                     # Actual number of springs per robot

        # --- Simulation state fields (need gradients for backprop) ---
        self.act = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.steps[None], self.max_n_springs[None]), needs_grad=self.needs_grad)                           # Neural net output: spring activation values per timestep
        self.x = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_sims[None], self.steps[None] + 1, self.max_n_masses[None]), needs_grad=self.needs_grad)                # Mass positions over time [sim, time, mass]
        self.center = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_sims[None], self.steps[None] + 1), needs_grad=self.needs_grad)                                    # Center of mass position per timestep
        self.v = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_sims[None], self.steps[None] + 1, self.max_n_masses[None]), needs_grad=self.needs_grad)                # Mass velocities over time
        self.vinc = ti.Vector.field(2, dtype=ti.f32, shape=(self.n_sims[None], self.steps[None] + 1, self.max_n_masses[None]), needs_grad=self.needs_grad)             # Velocity increments from spring impulses

        # --- Neural network weight fields ---
        # weights1: input->hidden layer. Input size = n_masses*4 features + CPG oscillators
        self.weights1 = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.max_n_masses[None] * 4 + self.nn_cpg_count[None], self.nn_hidden_size[None]), needs_grad=self.needs_grad)
        # weights2: hidden->output layer. Output size = one activation per spring
        self.weights2 = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.nn_hidden_size[None], self.max_n_springs[None]), needs_grad=self.needs_grad)
        self.biases1 = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.nn_hidden_size[None]), needs_grad=self.needs_grad)   # Biases for hidden layer
        self.biases2 = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.max_n_springs[None]), needs_grad=self.needs_grad)    # Biases for output layer

        # --- Adam optimizer moment fields (no gradients needed, these ARE the optimizer state) ---
        # m fields: 1st moment (exponential moving average of gradients)
        self.weights1_grad_m = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.max_n_masses[None] * 4 + self.nn_cpg_count[None], self.nn_hidden_size[None]), needs_grad=False)
        self.weights2_grad_m = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.nn_hidden_size[None], self.max_n_springs[None]), needs_grad=False)
        self.biases1_grad_m = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.nn_hidden_size[None]), needs_grad=False)
        self.biases2_grad_m = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.max_n_springs[None]), needs_grad=False)
        # v fields: 2nd moment (exponential moving average of squared gradients)
        self.weights1_grad_v = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.max_n_masses[None] * 4 + self.nn_cpg_count[None], self.nn_hidden_size[None]), needs_grad=False)
        self.weights2_grad_v = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.nn_hidden_size[None], self.max_n_springs[None]), needs_grad=False)
        self.biases1_grad_v = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.nn_hidden_size[None]), needs_grad=False)
        self.biases2_grad_v = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.max_n_springs[None]), needs_grad=False)

        # --- Height tracking fields ---
        self.max_height = ti.field(dtype=ti.f32, shape=(self.n_sims[None],), needs_grad=self.needs_grad)
        # sum_exp: intermediate log-sum-exp accumulator — needs grad so gradients flow back through to center
        self.sum_exp = ti.field(dtype=ti.f32, shape=(self.n_sims[None],), needs_grad=True)

        self.hidden = ti.field(dtype=ti.f32, shape=(self.n_sims[None], self.steps[None], self.nn_hidden_size[None]), needs_grad=self.needs_grad)  # Hidden layer activations at each timestep
        self.n_hidden = ti.field(dtype=ti.i32, shape=(self.n_sims[None],), needs_grad=False)                                                      # Actual hidden units used per robot (scales with n_masses)
        self.loss = ti.field(dtype=ti.f32, shape=(self.n_sims[None],), needs_grad=self.needs_grad)                                                 # Per-robot scalar loss value
        self.adam_step = ti.field(dtype=ti.i32, shape=(), needs_grad=False)                                                                        # Adam step counter (for bias correction)

    def train(self):
        fitness_history = []  # Track loss values across training iterations
        pbar = tqdm(total=self.config["learning_steps"], desc="Training")  # Initialize progress bar
        for i in range(self.config["learning_steps"]):
            fitness_history.append(self.learning_step())  # Run one forward+backward+update cycle
            pbar.update(1)  # Advance progress bar
        pbar.close()  # Clean up progress bar
        fitness_history.append(self.evaluation_step())  # Final eval run without gradient updates
        return -np.array(fitness_history).T  # Return negated loss as "fitness" (higher = better), shape [n_sims, steps]

    def learning_step(self):
        self.clear_grads()          # Zero all gradient buffers before each step
        self.reinitialize_robots()  # Reset all simulation state (positions, velocities, etc.)
        self.forward()              # Run full simulation forward (includes accumulate_sum_exp)
        self.compute_loss()         # Compute loss from accumulated sum_exp
        self.loss.grad.fill(1.0 / 20.0)  # Seed backprop: pre-scale by 1/temperature since the /temperature
                                         # division is applied in Python after compute_log_sum_exp, not inside the kernel
        self.backward()             # Backpropagate through all timesteps
        self.adam_step[None] += 1   # Increment Adam step counter for bias correction
        self.clip_grads()
        self.update_weights()       # Apply Adam update to all weights and biases
        return self.loss.to_numpy() # Return current loss values as numpy array

    def evaluation_step(self):
        self.reinitialize_robots()  # Reset simulation state
        self.forward()              # Run simulation forward only (no gradient tracking)
        self.compute_loss()         # Compute final loss
        return self.loss.to_numpy() # Return final fitness scores

    def forward(self):
        # Sequentially advance simulation one timestep at a time
        for t in range(0, self.steps[None]):
            self.compute_com(t)           # Compute center of mass at current timestep
            self.nn1(t)                   # Compute hidden layer activations from mass states
            self.nn2(t)                   # Compute spring activations from hidden layer
            self.apply_spring_force(t)    # Compute spring impulses and accumulate into vinc
            self.advance(t + 1)           # Integrate physics: update positions and velocities
            self.update_max_height(t + 1)
        self.compute_com(self.steps[None])  # Compute final center of mass after last step
        self.accumulate_sum_exp()           # Accumulate exp(height) across all timesteps into sum_exp

    def backward(self):
        # Replay forward operations in reverse order to accumulate gradients (BPTT)
        # Note: compute_loss is a Python wrapper (not a kernel), so we call compute_log_sum_exp.grad() directly.
        # The /temperature scaling was applied to loss values in Python — loss.grad is already seeded
        # with 1.0 in learning_step, which correctly propagates through the unscaled log kernel.
        self.compute_log_sum_exp.grad()                   # Backprop through -log(sum_exp) step
        self.accumulate_sum_exp.grad()                    # Backprop through sum_exp accumulation — gradients flow to self.center
        self.compute_com.grad(self.steps[None])           # Backprop through final COM computation
        for t in range(self.steps[None]-1, -1, -1):      # Iterate timesteps in reverse
            self.advance.grad(t + 1)                      # Backprop through physics integration
            self.apply_spring_force.grad(t)               # Backprop through spring force application
            self.nn2.grad(t)                              # Backprop through output layer
            self.nn1.grad(t)                              # Backprop through input layer
            self.compute_com.grad(t)                      # Backprop through COM computation

    def initialize(self, masses, springs):
        n_robots = len(masses)  # Number of robots to initialize
        assert n_robots == self.n_sims[None], "The number of robots does not match n_sims in the simulator config"
        self.hard_reset()  # Wipe all fields to zero before initializing new robots
        for i in range(n_robots):
            m = np.array(masses[i])  # Convert mass positions to numpy array
            assert m.shape[0] > 0, "The number of masses in a robot must be greater than 0"
            assert m.shape[0] <= self.max_n_masses[None], "The number of masses in a robot must be less than or equal to max_n_masses in the simulator config"
            m[:, 0] = m[:, 0] - m[:, 0].mean()                            # Center masses horizontally around x=0
            m[:, 1] = m[:, 1] - m[:, 1].min() + self.ground_height[None]  # Lift masses so lowest point sits on ground
            s = np.array(springs[i])  # Convert spring definitions to numpy array
            assert s.shape[0] > 0, "The number of springs in a robot must be greater than 0"
            assert s.shape[0] <= self.max_n_springs[None], "The number of springs in a robot must be less than or equal to max_n_springs in the simulator config"
            self.initialize_masses(i, m)    # Upload mass positions to Taichi field
            self.initialize_springs(i, s)   # Upload spring topology and compute rest lengths
        self.count_hidden_units()   # Scale hidden layer size proportionally to each robot's mass count
        self.initialize_weights()   # Initialize NN weights with He initialization

    def count_hidden_units(self):
        # Scale the number of active hidden units proportionally to how many masses this robot has
        # Robots with fewer masses use fewer hidden units (avoids wasteful computation)
        for sim_idx in range(self.n_sims[None]):
            self.n_hidden[sim_idx] = int(self.nn_hidden_size[None] * (self.n_masses[sim_idx] / self.max_n_masses[None]))

    @ti.kernel
    def initialize_masses(self, i: ti.i32, masses: ti.types.ndarray()):
        for j in range(masses.shape[0]):
            self.x[i, 0, j] = ti.Vector([masses[j, 0], masses[j, 1]], dt=ti.f32)  # Set initial position of each mass
            self.n_masses[i] += 1  # Increment mass count for this robot

    @ti.kernel
    def initialize_springs(self, i: ti.i32, springs: ti.types.ndarray()):
        for j in range(springs.shape[0]):
            self.springs[i, j] = ti.Vector([springs[j, 0], springs[j, 1]], dt=ti.i32)                               # Store endpoint mass indices for this spring
            self.springL[i, j] = ti.math.distance(self.x[i, 0, springs[j, 0]], self.x[i, 0, springs[j, 1]])        # Compute and store rest length from initial mass positions
            self.n_springs[i] += 1  # Increment spring count for this robot

    def initialize_weights(self):
        weights1 = []  # Collect weight arrays for all robots before uploading
        weights2 = []
        for i in range(self.n_sims[None]):
            fan_in1 = self.n_masses[i] * 4 + self.nn_cpg_count[None]  # Input fan-in: 4 features per mass + CPG count
            # He initialization: variance = 2/fan_in, good for tanh/relu networks
            weights1.append(np.random.normal(0.0, np.sqrt(2.0 / fan_in1), (self.max_n_masses[None] * 4 + self.nn_cpg_count[None], self.nn_hidden_size[None])))
            fan_in2 = self.n_hidden[i]  # Hidden layer fan-in
            weights2.append(np.random.normal(0.0, np.sqrt(2.0 / fan_in2), (self.nn_hidden_size[None], self.max_n_springs[None])))
        self.weights1.from_numpy(np.stack(weights1, dtype=np.float32))   # Upload all robots' weights1 in one batch
        self.weights2.from_numpy(np.stack(weights2, dtype=np.float32))   # Upload all robots' weights2 in one batch
        self.biases1.from_numpy(np.zeros((self.n_sims[None], self.nn_hidden_size[None]), dtype=np.float32))      # Initialize biases to zero
        self.biases2.from_numpy(np.zeros((self.n_sims[None], self.max_n_springs[None]), dtype=np.float32))       # Initialize biases to zero

    @ti.kernel
    def nn1(self, t: ti.i32):
        # --- Mass state contributions to hidden layer ---
        for sim_idx, mass_idx, hidden_idx in ti.ndrange(self.n_sims[None], self.max_n_masses[None], self.nn_hidden_size[None]):
            if mass_idx < self.n_masses[sim_idx] and hidden_idx < self.n_hidden[sim_idx]:
                # Feature 0: x-velocity scaled down by 0.05 to normalize magnitude
                self.hidden[sim_idx, t, hidden_idx] += self.weights1[sim_idx, mass_idx * 4 + 0, hidden_idx] * self.v[sim_idx, t, mass_idx][0] * 0.05
                # Feature 1: y-velocity scaled down by 0.05
                self.hidden[sim_idx, t, hidden_idx] += self.weights1[sim_idx, mass_idx * 4 + 1, hidden_idx] * self.v[sim_idx, t, mass_idx][1] * 0.05
                # Feature 2: x-offset of mass from center of mass (proprioception)
                self.hidden[sim_idx, t, hidden_idx] += self.weights1[sim_idx, mass_idx * 4 + 2, hidden_idx] * (self.center[sim_idx, t].x - self.x[sim_idx, t, mass_idx].x)
                # Feature 3: y-offset of mass from center of mass
                self.hidden[sim_idx, t, hidden_idx] += self.weights1[sim_idx, mass_idx * 4 + 3, hidden_idx] * (self.center[sim_idx, t].y - self.x[sim_idx, t, mass_idx].y)
        # --- CPG oscillator contributions to hidden layer ---
        for sim_idx, cpg_idx, hidden_idx in ti.ndrange(self.n_sims[None], self.nn_cpg_count[None], self.nn_hidden_size[None]):
            if hidden_idx < self.n_hidden[sim_idx]:
                # Each CPG neuron is a sinusoid with evenly spaced phase offset, injecting rhythm into the network
                self.hidden[sim_idx, t, hidden_idx] += self.weights1[sim_idx, self.max_n_masses[None] * 4 + cpg_idx, hidden_idx] * ti.math.sin(t * self.dt[None] * self.cpg_omega[None] + (cpg_idx * 2 * ti.math.pi / self.nn_cpg_count[None]))
        # --- Add biases to complete hidden layer computation ---
        for sim_idx, hidden_idx in ti.ndrange(self.n_sims[None], self.nn_hidden_size[None]):
            if hidden_idx < self.n_hidden[sim_idx]:
                self.hidden[sim_idx, t, hidden_idx] += self.biases1[sim_idx, hidden_idx]  # Add learned bias to each hidden unit

    @ti.kernel
    def nn2(self, t: ti.i32):
        # --- Project hidden layer to spring activations ---
        for sim_idx, hidden_idx, spring_idx in ti.ndrange(self.n_sims[None], self.nn_hidden_size[None], self.max_n_springs[None]):
            if spring_idx < self.n_springs[sim_idx] and hidden_idx < self.n_hidden[sim_idx]:
                # Apply tanh nonlinearity to hidden unit, then multiply by weight and accumulate into spring activation
                self.act[sim_idx, t, spring_idx] += self.weights2[sim_idx, hidden_idx, spring_idx] * ti.math.tanh(self.hidden[sim_idx, t, hidden_idx])
        # --- Add output biases ---
        for sim_idx, spring_idx in ti.ndrange(self.n_sims[None], self.max_n_springs[None]):
            if spring_idx < self.n_springs[sim_idx]:
                self.act[sim_idx, t, spring_idx] += self.biases2[sim_idx, spring_idx]  # Add learned bias to each spring activation

    @ti.kernel
    def apply_spring_force(self, t: ti.i32):
        for sim_idx, spring_idx in ti.ndrange(self.n_sims[None], self.max_n_springs[None]):
            if spring_idx < self.n_springs[sim_idx]:
                endpoint1 = self.springs[sim_idx, spring_idx][0]  # Index of first mass connected to this spring
                endpoint2 = self.springs[sim_idx, spring_idx][1]  # Index of second mass connected to this spring
                dist = self.x[sim_idx, t, endpoint1] - self.x[sim_idx, t, endpoint2]  # Vector from mass2 to mass1
                length = dist.norm()  # Current spring length (scalar distance)
                # Target length is rest length modulated by neural net output via tanh (bounded actuation)
                target_length = self.springL[sim_idx, spring_idx] * (1 + ti.math.tanh(self.act[sim_idx, t, spring_idx]) * self.springA[None])
                # Hooke's law: force proportional to stretch, directed along spring axis
                force = (length - target_length) * self.springK[None] * dist / (length + self.eps[None])
                impulse = self.dt[None] * force  # Convert force to velocity impulse (F*dt = dp)
                self.vinc[sim_idx, t+1, endpoint1] += -impulse  # Equal and opposite impulse to mass1
                self.vinc[sim_idx, t+1, endpoint2] += impulse   # Equal and opposite impulse to mass2

    @ti.func
    def get_terrain_info(self, x_pos, y_pos):
        # Parameters for the terrain
        gap_half, elev, slope, slant, thick = 0.25, self.rung_elev[None], 0.2, 1.0, 0.05
        f_h, c_h = self.ground_height[None], 100.0
        
        # Radius of the "rounding" effect
        rounding_radius = 0.15 

        for i in ti.static(range(1,20)):
            row_y = self.ground_height[None] + (ti.cast(i, ti.f32) * elev)
            dx = ti.abs(x_pos)
            
            if dx > gap_half and dx < (gap_half + slant):
                top = row_y + ((dx - gap_half) * slope)
                
                # --- ROUNDED CEILING LOGIC ---
                # Standard flat bottom
                bot = top - thick 
                
                # If the mass is near the inner edge (the gap), 
                # we curve the bottom height 'upward'
                dist_from_edge = dx - gap_half
                if dist_from_edge < rounding_radius:
                    # Circular arc formula: creates a smooth curve up to the edge
                    # This makes the "corner" of the ceiling feel like a ball
                    offset = rounding_radius - ti.sqrt(ti.max(0.0, rounding_radius**2 - (rounding_radius - dist_from_edge)**2))
                    bot += offset

                if y_pos >= top - 0.01: 
                    f_h = ti.max(f_h, top)
                elif y_pos <= bot + 0.01: 
                    c_h = ti.min(c_h, bot)
                    
        return f_h, c_h

    @ti.kernel
    def advance(self, t: ti.i32):
        for sim_idx, mass_idx in ti.ndrange(self.n_sims[None], self.max_n_masses[None]):
            if mass_idx < self.n_masses[sim_idx]:
                damping = ti.exp(-self.dt[None] * self.drag_damping[None])
                g = self.dt[None] * ti.Vector([0.0, -self.gravity[None]])
                v_o, x_o = self.v[sim_idx, t-1, mass_idx], self.x[sim_idx, t-1, mass_idx]
                new_v = damping * v_o + g + self.vinc[sim_idx, t, mass_idx]
                new_x = x_o + self.dt[None] * new_v
                
                # Check for floor and ceiling rungs
                f_h, c_h = self.get_terrain_info(new_x[0], x_o[1])
                if new_x[1] < f_h: 
                    new_x[1], new_v.y = f_h, 0.0
                elif new_x[1] > c_h: 
                    new_x[1], new_v.y = c_h, 0.0
                
                self.x[sim_idx, t, mass_idx], self.v[sim_idx, t, mass_idx] = new_x, new_v

    @ti.func
    def v_on_contact(self, v_old: vec2, normal: vec2) -> vec2:
        vn = v_old.dot(normal) * normal  # Normal component of velocity (perpendicular to ground, pointing into it)
        vn_mag = vn.norm()               # Magnitude of normal velocity (impact speed)
        vt = v_old - vn                  # Tangential component of velocity (sliding along ground)
        vnew = self.restitution[None] * -vn  # Bounce: flip normal velocity and scale by restitution coefficient
        vt_mag = vt.norm()               # Speed of sliding
        if vt_mag > 0.0:
            # Coulomb friction: friction force proportional to normal force, opposing slide direction
            friction_mag = ti.math.clamp(self.friction[None] * vn_mag, 0.0, vt_mag * 0.95)  # Clamped to 95% of slide speed to prevent reversal
            vf = -friction_mag * vt.normalized()  # Friction impulse opposing sliding direction
            vnew += vt + vf  # Add tangential velocity reduced by friction
        return vnew  # Return post-contact velocity

    @ti.kernel
    def compute_com(self, t: ti.i32):
        for sim_idx, mass_idx in ti.ndrange(self.n_sims[None], self.max_n_masses[None]):
            if mass_idx < self.n_masses[sim_idx]:
                # Accumulate each mass's contribution to the center of mass (average position)
                self.center[sim_idx, t] += self.x[sim_idx, t, mass_idx] / ti.cast(self.n_masses[sim_idx], ti.f32)

    @ti.kernel
    def accumulate_sum_exp(self):
        # Accumulate exp(height * TEMPERATURE) across all timesteps using a flat 2D loop.
        # Taichi autodiff requires a single flat parallel loop per kernel — nested loops are not supported.
        # TEMPERATURE is a module-level constant (not a local variable) to avoid the autodiff restriction
        # that prohibits mixing scalar assignments with for-loops in the same kernel.
        for sim_idx, t in ti.ndrange(self.n_sims[None], self.steps[None] + 1):
            delta_y = self.center[sim_idx, t].y - self.center[sim_idx, 0].y
            self.sum_exp[sim_idx] += ti.exp(delta_y * TEMPERATURE)

    @ti.kernel
    def compute_log_sum_exp(self):
        # Take log of accumulated sum_exp — kept as its own kernel because Taichi autodiff
        # cannot mix local scalar assignments (temperature = 20.0) with for-loops in one kernel.
        for sim_idx in range(self.n_sims[None]):
            self.loss[sim_idx] = -ti.log(self.sum_exp[sim_idx])

    def compute_loss(self):
        # Wrapper that runs the full log-sum-exp loss in two autodiff-compatible steps.
        # The /TEMPERATURE division happens here in Python (not inside a kernel) to avoid the
        # "mixed loop and non-loop statements" autodiff restriction on local scalar assignments.
        self.compute_log_sum_exp()
        self.loss.from_numpy(self.loss.to_numpy() / TEMPERATURE)

    def clip_grads(self, clip_value=1.0):
        # Replace NaN/Inf gradients with 0 first, then clip to [-clip_value, clip_value].
        # np.clip alone does NOT fix NaNs — NaN comparisons always return False so NaNs pass through unchanged.
        def sanitize(arr):
            arr = np.nan_to_num(arr, nan=0.0, posinf=clip_value, neginf=-clip_value)
            return np.clip(arr, -clip_value, clip_value)

        self.weights1.grad.from_numpy(sanitize(self.weights1.grad.to_numpy()))
        self.weights2.grad.from_numpy(sanitize(self.weights2.grad.to_numpy()))
        self.biases1.grad.from_numpy(sanitize(self.biases1.grad.to_numpy()))
        self.biases2.grad.from_numpy(sanitize(self.biases2.grad.to_numpy()))


    @ti.kernel
    def old_compute_loss(self):
        for sim_idx in range(self.n_sims[None]):
            com0 = self.center[sim_idx, 0].x              # Starting x-position of center of mass
            comt = self.center[sim_idx, self.steps[None]].x  # Final x-position of center of mass
            self.loss[sim_idx] = com0 - comt              # Loss = negative displacement (minimizing loss = maximizing rightward movement)

    @ti.kernel
    def update_weights(self):
        # --- Adam update for weights1 ---
        for sim_idx, i, j in ti.ndrange(self.n_sims[None], self.max_n_masses[None] * 4 + self.nn_cpg_count[None], self.nn_hidden_size[None]):
            grad = self.weights1.grad[sim_idx, i, j]  # Gradient from backprop
            # Update biased 1st moment estimate (running mean of gradients)
            self.weights1_grad_m[sim_idx, i, j] = self.adam_beta1[None] * self.weights1_grad_m[sim_idx, i, j] + (1.0 - self.adam_beta1[None]) * grad
            # Update biased 2nd moment estimate (running mean of squared gradients)
            self.weights1_grad_v[sim_idx, i, j] = self.adam_beta2[None] * self.weights1_grad_v[sim_idx, i, j] + (1.0 - self.adam_beta2[None]) * grad * grad
            # Bias correction: compensate for zero-initialization of moments at early steps
            m_hat = self.weights1_grad_m[sim_idx, i, j] / (1.0 - ti.pow(self.adam_beta1[None], self.adam_step[None]))
            v_hat = self.weights1_grad_v[sim_idx, i, j] / (1.0 - ti.pow(self.adam_beta2[None], self.adam_step[None]))
            # Adam update: step size is lr * m_hat / (sqrt(v_hat) + eps), adapts per-parameter
            self.weights1[sim_idx, i, j] += -self.learning_rate[None] * m_hat / (ti.sqrt(v_hat) + self.eps[None])
        # --- Adam update for weights2 ---
        for sim_idx, i, j in ti.ndrange(self.n_sims[None], self.nn_hidden_size[None], self.max_n_springs[None]):
            grad = self.weights2.grad[sim_idx, i, j]
            self.weights2_grad_m[sim_idx, i, j] = self.adam_beta1[None] * self.weights2_grad_m[sim_idx, i, j] + (1.0 - self.adam_beta1[None]) * grad
            self.weights2_grad_v[sim_idx, i, j] = self.adam_beta2[None] * self.weights2_grad_v[sim_idx, i, j] + (1.0 - self.adam_beta2[None]) * grad * grad
            m_hat = self.weights2_grad_m[sim_idx, i, j] / (1.0 - ti.pow(self.adam_beta1[None], self.adam_step[None]))
            v_hat = self.weights2_grad_v[sim_idx, i, j] / (1.0 - ti.pow(self.adam_beta2[None], self.adam_step[None]))
            self.weights2[sim_idx, i, j] += -self.learning_rate[None] * m_hat / (ti.sqrt(v_hat) + self.eps[None])
        # --- Adam update for biases1 ---
        for sim_idx, i in ti.ndrange(self.n_sims[None], self.nn_hidden_size[None]):
            grad = self.biases1.grad[sim_idx, i]
            self.biases1_grad_m[sim_idx, i] = self.adam_beta1[None] * self.biases1_grad_m[sim_idx, i] + (1.0 - self.adam_beta1[None]) * grad
            self.biases1_grad_v[sim_idx, i] = self.adam_beta2[None] * self.biases1_grad_v[sim_idx, i] + (1.0 - self.adam_beta2[None]) * grad * grad
            m_hat = self.biases1_grad_m[sim_idx, i] / (1.0 - ti.pow(self.adam_beta1[None], self.adam_step[None]))
            v_hat = self.biases1_grad_v[sim_idx, i] / (1.0 - ti.pow(self.adam_beta2[None], self.adam_step[None]))
            self.biases1[sim_idx, i] += -self.learning_rate[None] * m_hat / (ti.sqrt(v_hat) + self.eps[None])
        # --- Adam update for biases2 ---
        for sim_idx, i in ti.ndrange(self.n_sims[None], self.max_n_springs[None]):
            grad = self.biases2.grad[sim_idx, i]
            self.biases2_grad_m[sim_idx, i] = self.adam_beta1[None] * self.biases2_grad_m[sim_idx, i] + (1.0 - self.adam_beta1[None]) * grad
            self.biases2_grad_v[sim_idx, i] = self.adam_beta2[None] * self.biases2_grad_v[sim_idx, i] + (1.0 - self.adam_beta2[None]) * grad * grad
            m_hat = self.biases2_grad_m[sim_idx, i] / (1.0 - ti.pow(self.adam_beta1[None], self.adam_step[None]))
            v_hat = self.biases2_grad_v[sim_idx, i] / (1.0 - ti.pow(self.adam_beta2[None], self.adam_step[None]))
            self.biases2[sim_idx, i] += -self.learning_rate[None] * m_hat / (ti.sqrt(v_hat) + self.eps[None])

    @ti.kernel
    def reinitialize_robots(self):
        # Reset mass positions to initial state (t=0 is kept, t>0 cleared to zero)
        for sim_idx, t, mass_idx in ti.ndrange(self.n_sims[None], self.steps[None] + 1, self.max_n_masses[None]):
            if t > 0:
                self.x[sim_idx, t, mass_idx] = ti.Vector([0.0, 0.0], dt=ti.f32)  # Clear all non-initial positions
        # Reset velocities, velocity increments, and center of mass for all timesteps
        for sim_idx, t, mass_idx in ti.ndrange(self.n_sims[None], self.steps[None] + 1, self.max_n_masses[None]):
            self.v[sim_idx, t, mass_idx] = ti.Vector([0.0, 0.0], dt=ti.f32)      # Zero all velocities
            self.vinc[sim_idx, t, mass_idx] = ti.Vector([0.0, 0.0], dt=ti.f32)   # Zero all spring impulse accumulators
            self.center[sim_idx, t] = ti.Vector([0.0, 0.0], dt=ti.f32)           # Zero all COM positions
        # Reset spring activations
        for sim_idx, t, spring_idx in ti.ndrange(self.n_sims[None], self.steps[None], self.max_n_springs[None]):
            self.act[sim_idx, t, spring_idx] = 0.0  # Zero neural net spring outputs
        # Reset per-sim loss and intermediate fields
        for sim_idx in range(self.n_sims[None]):
            self.loss[sim_idx] = 0.0       # Clear loss accumulator
            self.sum_exp[sim_idx] = 0.0    # Clear log-sum-exp accumulator so it doesn't carry over between steps
            self.max_height[sim_idx] = 0.0 # Clear max height tracker
        # Reset hidden layer activations
        for sim_idx, t, hidden_idx in ti.ndrange(self.n_sims[None], self.steps[None], self.nn_hidden_size[None]):
            self.hidden[sim_idx, t, hidden_idx] = 0.0  # Zero all hidden unit activations

    def clear_grads(self):
        # Zero all gradient buffers before each training step to prevent gradient accumulation
        self.x.grad.fill(0.0)         # Position gradients
        self.center.grad.fill(0.0)    # COM gradients
        self.v.grad.fill(0.0)         # Velocity gradients
        self.vinc.grad.fill(0.0)      # Spring impulse gradients
        self.act.grad.fill(0.0)       # Spring activation gradients
        self.loss.grad.fill(0.0)      # Loss gradients
        self.sum_exp.grad.fill(0.0)   # log-sum-exp intermediate gradients — must clear or they accumulate across steps
        self.weights1.grad.fill(0.0)  # Input layer weight gradients
        self.weights2.grad.fill(0.0)  # Output layer weight gradients
        self.biases1.grad.fill(0.0)   # Hidden bias gradients
        self.biases2.grad.fill(0.0)   # Output bias gradients
        self.hidden.grad.fill(0.0)    # Hidden activation gradients
        self.max_height.grad.fill(0.0) # Max height gradients

    def hard_reset(self):
        # Full reset of ALL fields (called when loading new robot morphologies)
        self.x.fill(0.0)              # Clear all positions
        self.center.fill(0.0)         # Clear all COM positions
        self.v.fill(0.0)              # Clear all velocities
        self.vinc.fill(0.0)           # Clear all spring impulses
        self.n_masses.fill(0)         # Reset mass counts
        self.springL.fill(0.0)        # Clear spring rest lengths
        self.springs.fill(0)          # Clear spring topology
        self.n_springs.fill(0)        # Reset spring counts
        self.act.fill(0.0)            # Clear spring activations
        self.loss.fill(0.0)           # Clear losses
        self.sum_exp.fill(0.0)        # Clear log-sum-exp accumulators
        self.max_height.fill(0.0)     # Clear max height trackers
        self.weights1.fill(0.0)       # Clear input layer weights
        self.weights2.fill(0.0)       # Clear output layer weights
        self.biases1.fill(0.0)        # Clear hidden biases
        self.biases2.fill(0.0)        # Clear output biases
        self.weights1_grad_m.fill(0.0)  # Clear Adam 1st moment for weights1
        self.weights2_grad_m.fill(0.0)  # Clear Adam 1st moment for weights2
        self.biases1_grad_m.fill(0.0)   # Clear Adam 1st moment for biases1
        self.biases2_grad_m.fill(0.0)   # Clear Adam 1st moment for biases2
        self.weights1_grad_v.fill(0.0)  # Clear Adam 2nd moment for weights1
        self.weights2_grad_v.fill(0.0)  # Clear Adam 2nd moment for weights2
        self.biases1_grad_v.fill(0.0)   # Clear Adam 2nd moment for biases1
        self.biases2_grad_v.fill(0.0)   # Clear Adam 2nd moment for biases2
        self.hidden.fill(0.0)           # Clear hidden activations
        self.n_hidden.fill(0)           # Reset hidden unit counts
        self.adam_step[None] = 0        # Reset Adam step counter
        if self.needs_grad:
            self.clear_grads()          # Also clear gradient buffers if training mode

    def get_control_params(self, sim_idx):
        # Export neural network weights for specified robot indices (e.g. to save or transfer to new sim)
        params = []
        weights1 = self.weights1.to_numpy()  # Copy all weights1 from GPU to CPU
        weights2 = self.weights2.to_numpy()  # Copy all weights2 from GPU to CPU
        biases1 = self.biases1.to_numpy()    # Copy all biases1 from GPU to CPU
        biases2 = self.biases2.to_numpy()    # Copy all biases2 from GPU to CPU
        for i in sim_idx:
            w1 = weights1[i]  # Extract weights for robot i
            w2 = weights2[i]
            b1 = biases1[i]
            b2 = biases2[i]
            params.append(
                {
                    "weights1": w1,
                    "weights2": w2,
                    "biases1": b1,
                    "biases2": b2,
                }
            )
        return params  # Return list of weight dicts, one per requested robot

    def set_control_params(self, sim_idx, control_params):
        # Import neural network weights for specified robots (e.g. to resume training or evaluate saved policy)
        weights1 = self.weights1.to_numpy()  # Pull current weights to CPU to modify
        weights2 = self.weights2.to_numpy()
        biases1 = self.biases1.to_numpy()
        biases2 = self.biases2.to_numpy()
        for idx, i in enumerate(sim_idx):
            w1 = control_params[idx]["weights1"]  # Get weights for robot at position idx in input list
            w2 = control_params[idx]["weights2"]
            b1 = control_params[idx]["biases1"]
            b2 = control_params[idx]["biases2"]
            weights1[i] = w1  # Overwrite weights for robot i
            weights2[i] = w2
            biases1[i] = b1
            biases2[i] = b2
        self.weights1.from_numpy(weights1)  # Push all updated weights back to GPU in one transfer
        self.weights2.from_numpy(weights2)
        self.biases1.from_numpy(biases1)
        self.biases2.from_numpy(biases2)

    @ti.kernel
    def update_max_height(self, t: ti.i32):
        for sim_idx in range(self.n_sims[None]):
            self.max_height[sim_idx] = ti.max(
                self.max_height[sim_idx],
                self.center[sim_idx, t].y
            )