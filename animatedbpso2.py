import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
import matplotlib as mpl
from itertools import product

# ========================================
# TUNABLE BPSO PARAMETERS
# ========================================
class BPSOConfig:
    """Configuration class for Binary PSO parameters"""
    def __init__(self, w_initial=1.3, w_final=0.135, c1=1.4, c2=1.6, 
                 transfer_func='v_shape', grid_size=7):
        # Grid parameters
        self.grid_size = grid_size
        self.n_particles = 100  # For visualization, we show one particle's evolution
        
        # PSO parameters
        self.max_iterations = 40
        
        self.w_initial = w_initial  # Initial inertia weight
        self.w_final = w_final     # Final inertia weight
        self.c1 = c1       # Cognitive coefficient (personal best attraction)
        self.c2 = c2     # Social coefficient (global best attraction)
        
        # Velocity parameters
        self.v_max = 6.0         # Maximum velocity (for clamping)
        self.v_min = -6.0        # Minimum velocity (for clamping)
        
        # Transfer function type
        self.transfer_func = transfer_func  # Options: 'sigmoid', 'tanh', 'v_shape'
        
        # Animation parameters
        self.animation_interval = 15  # milliseconds between frames
        self.save_animation = False
        self.animation_filename = 'bpso_animation.gif'
        
        # Fitness function parameters
        self.target_pattern = 'checkerboard'  # Options: 'checkerboard', 'cross', 'border', 'random'
        
    def get_inertia_weight(self, iteration):
        """Calculate inertia weight with linear decrease"""
        return self.w_initial - (self.w_initial - self.w_final) * iteration / self.max_iterations

# # Define parameter ranges to test
# w_initial_list = [1.4, 1.2, 1.0]
# w_final_list = [0.1, 0.2, 0.05]
# c1_list = [1.5, 1.2, 1.8]
# c2_list = [1.5, 1.2, 1.8]
# transfer_func_list = ['v_shape', 'sigmoid', 'tanh']
# grid_size_list = [7, 9, 5]

# # Create all combinations (this will generate many configurations)
# configurations = []
# for params in product(w_initial_list, w_final_list, c1_list, c2_list, transfer_func_list, grid_size_list):
#     config = BPSOConfig(w_initial=params[0], w_final=params[1], 
#                         c1=params[2], c2=params[3], 
#                         transfer_func=params[4], grid_size=params[5])
#     configurations.append(config)

# Or select specific configurations you want to compare
configurations = [
    BPSOConfig(w_initial=0.9, w_final=0.1, c1=1.4, c2=1.6, transfer_func='v_shape', grid_size=8),

]


# configurations = [
#     BPSOConfig(w_initial=1.1, w_final=0.3, c1=0.9, c2=2.1, transfer_func='v_shape', grid_size=15),
#     # BPSOConfig(w_initial=1.1, w_final=0.3, c1=1.0, c2=2.0, transfer_func='v_shape', grid_size=15),
#     BPSOConfig(w_initial=1.1, w_final=0.3, c1=1.1, c2=1.9, transfer_func='v_shape', grid_size=15),
#     # BPSOConfig(w_initial=1.1, w_final=0.3, c1=1.2, c2=1.8, transfer_func='v_shape', grid_size=15),
#     BPSOConfig(w_initial=1.1, w_final=0.3, c1=1.3, c2=1.7, transfer_func='v_shape', grid_size=15),
#     # BPSOConfig(w_initial=1.1, w_final=0.3, c1=1.4, c2=1.6, transfer_func='v_shape', grid_size=15),    
#     BPSOConfig(w_initial=1.1, w_final=0.3, c1=1.49, c2=1.49, transfer_func='v_shape', grid_size=15),
#     # BPSOConfig(w_initial=1.1, w_final=0.3, c1=1.7, c2=1.3, transfer_func='v_shape', grid_size=15),
#     BPSOConfig(w_initial=1.1, w_final=0.3, c1=1.8, c2=1.2, transfer_func='v_shape', grid_size=15),
#     # BPSOConfig(w_initial=1.1, w_final=0.3, c1=1.9, c2=1.1, transfer_func='v_shape', grid_size=15),
#     BPSOConfig(w_initial=1.1, w_final=0.3, c1=2.0, c2=1.0, transfer_func='v_shape', grid_size=15),
#     # BPSOConfig(w_initial=1.1, w_final=0.3, c1=2.1, c2=0.9, transfer_func='v_shape', grid_size=15),


# ]



# ========================================
# FITNESS FUNCTIONS
# ========================================
def create_target_pattern(pattern_type, size):
    """Create different target patterns for optimization"""
    if pattern_type == 'checkerboard':
        pattern = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                pattern[i, j] = (i + j) % 2
        return pattern
    
    elif pattern_type == 'cross':
        pattern = np.zeros((size, size))
        mid = size // 2
        pattern[mid, :] = 1  # Horizontal line
        pattern[:, mid] = 1  # Vertical line
        return pattern
    
    elif pattern_type == 'border':
        pattern = np.zeros((size, size))
        pattern[0, :] = 1    # Top border
        pattern[-1, :] = 1   # Bottom border
        pattern[:, 0] = 1    # Left border
        pattern[:, -1] = 1   # Right border
        return pattern
    
    elif pattern_type == 'random':
        np.random.seed(42)  # Fixed seed for reproducibility
        return np.random.randint(0, 2, (size, size))
    
    else:
        return np.ones((size, size)) * 0.5  # Default pattern

def fitness_function(position, target):
    """Calculate fitness as similarity to target pattern"""
    # Fitness is the number of matching pixels (higher is better)
    return np.sum(position == target)

# ========================================
# TRANSFER FUNCTIONS
# ========================================
def transfer_function(velocity, func_type='sigmoid'):
    """Apply transfer function to convert velocity to probability"""
    if func_type == 'sigmoid':
        return 1.0 / (1.0 + np.exp(-velocity))
    elif func_type == 'tanh':
        return (np.tanh(velocity) + 1) / 2
    elif func_type == 'v_shape':
        return np.abs(np.tanh(velocity))
    else:
        return 1.0 / (1.0 + np.exp(-velocity))  # Default to sigmoid

# ========================================
# BINARY PSO CLASS
# ========================================
class BinaryPSO:
    def __init__(self, config):
        self.config = config
        self.size = config.grid_size
        
        # Create target pattern
        self.target = create_target_pattern(config.target_pattern, self.size)
        
        # Initialize particle position (binary matrix)
        self.position = np.random.randint(0, 2, (self.size, self.size))
        
        # Initialize velocity (continuous values)
        self.velocity = np.random.uniform(-1, 1, (self.size, self.size))
        
        # Personal best
        self.personal_best_position = self.position.copy()
        self.personal_best_fitness = fitness_function(self.position, self.target)
        
        # Global best (in this case, same as personal best since we have one particle)
        self.global_best_position = self.personal_best_position.copy()
        self.global_best_fitness = self.personal_best_fitness
        
        # History tracking
        self.fitness_history = [self.personal_best_fitness]
        self.flip_history = []
        self.velocity_history = []
        self.inertia_history = []
        
    def update_particle(self, iteration):
        """Update particle position and velocity according to BPSO rules"""
        # Calculate inertia weight
        w = self.config.get_inertia_weight(iteration)
        
        # Generate random coefficients for cognitive and social components
        r1 = np.random.rand(self.size, self.size)
        r2 = np.random.rand(self.size, self.size)
        
        # Update velocity using BPSO equation
        cognitive_component = self.config.c1 * r1 * (self.personal_best_position - self.position)
        social_component = self.config.c2 * r2 * (self.global_best_position - self.position)
        
        self.velocity = (w * self.velocity + 
                        cognitive_component + 
                        social_component)
        
        # Clamp velocity to prevent explosion
        self.velocity = np.clip(self.velocity, self.config.v_min, self.config.v_max)
        
        # Apply transfer function to get flip probabilities
        flip_probabilities = transfer_function(self.velocity, self.config.transfer_func)
        
        # Generate random numbers for position update
        random_matrix = np.random.rand(self.size, self.size)
        
        # Update position (flip bits based on probabilities)
        old_position = self.position.copy()
        flip_mask = random_matrix < flip_probabilities
        self.position = np.where(flip_mask, 1 - self.position, self.position)
        
        # Count flips
        num_flips = np.sum(old_position != self.position)
        
        # Calculate fitness
        current_fitness = fitness_function(self.position, self.target)
        
        # Update personal best
        if current_fitness > self.personal_best_fitness:
            self.personal_best_position = self.position.copy()
            self.personal_best_fitness = current_fitness
            
            # Update global best (same as personal best for single particle)
            self.global_best_position = self.personal_best_position.copy()
            self.global_best_fitness = self.personal_best_fitness
        
        # Store history
        self.fitness_history.append(current_fitness)
        self.flip_history.append(num_flips)
        self.velocity_history.append(np.mean(np.abs(self.velocity)))
        self.inertia_history.append(w)
        
        return {
            'position': self.position,
            'fitness': current_fitness,
            'num_flips': num_flips,
            'avg_velocity': np.mean(np.abs(self.velocity)),
            'flip_probability': np.mean(flip_probabilities),
            'inertia': w,
            'best_fitness': self.personal_best_fitness
        }

# ========================================
# RUN OPTIMIZATION FOR ALL CONFIGURATIONS
# ========================================
all_flip_histories = []
all_configs = []

for config in configurations:
    pso = BinaryPSO(config)
    
    # Run optimization
    for iteration in range(config.max_iterations):
        pso.update_particle(iteration)
    
    # Store results
    all_flip_histories.append(pso.flip_history)
    all_configs.append(config)

# ========================================
# VISUALIZATION OF FLIP HISTORIES
# ========================================
plt.figure(figsize=(12, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, len(all_flip_histories)))

for i, (flips, config) in enumerate(zip(all_flip_histories, all_configs)):
    label = (f"w_i={config.w_initial}, w_f={config.w_final}, "
             f"c1={config.c1}, c2={config.c2}, "
             f"tf={config.transfer_func}, grid={config.grid_size}")
    plt.plot(range(len(flips)), flips, color=colors[i], label=label)

plt.xlabel("Iteration")
plt.ylabel("Number of Flips")
plt.title("Flip Statistics for Different Configurations")
plt.grid(True, linestyle='--', alpha=0.9)

# Adjust legend position and size
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='medium')

plt.tight_layout()
plt.show()

# Print summary of results
print("\nOptimization Results Summary:")
for i, (flips, config) in enumerate(zip(all_flip_histories, all_configs)):
    print(f"\nConfiguration {i+1}:")
    print(f"  Parameters: w_i={config.w_initial}, w_f={config.w_final}, "
          f"c1={config.c1}, c2={config.c2}, tf={config.transfer_func}, grid={config.grid_size}")
    print(f"  Final flips: {flips[-1] if flips else 'N/A'}")
    print(f"  Average flips: {np.mean(flips) if flips else 'N/A':.1f}")