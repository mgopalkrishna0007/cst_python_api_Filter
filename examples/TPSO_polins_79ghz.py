import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import time
import shutil
import sys
from datetime import datetime
import warnings
import context
import cst_python_api as cpa
import win32com.client
from win32com.client import gencache
import logging
import pickle
import json

# Set up logging
log_dir = os.path.join(os.getcwd(), "logfiles")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"optimization_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = Logger(log_file)
print(f"=== Optimization started at {datetime.now()} ===")

# Global variables for storing results
results_storage = {
    'below_threshold': [],
    'good_solutions': [],
    'best_solution': None,
    'all_iterations': []
}

# Global list to store all particles data
all_particles_data = []

inputparameters = [
    79,  # center frequency (GHz)
    8,  # bandwidth (GHz)
    2,  # dimension of unit cell (mm)
    0.8,  # width of pixel (mm)
    8,  # number of pixels (npix)
    0.0000000001,  # target mean squared error (MSE)
    0  # substrate type index
]

class TPSOConfig:
    """Configuration class for Ternary PSO parameters"""
    def __init__(self):
        # PSO parameters
        self.n_particles = 100
        self.max_iterations = 20
        self.w_initial = 1.1  # Initial inertia weight
        self.w_final = 0.1  # Final inertia weight
        self.c1 = 2  # Cognitive coefficient
        self.c2 = 2  # Social coefficient
        self.v_max = 6.0  # Maximum velocity
        self.v_min = -6.0  # Minimum velocity
        self.mse_threshold = inputparameters[5]

    def get_inertia_weight(self, iteration):
        """Calculate inertia weight with linear decrease"""
        return self.w_initial - (self.w_initial - self.w_final) * (iteration / self.max_iterations)

def clear_com_cache():
    try:
        temp_gen_py = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Temp', 'gen_py')
        if os.path.exists(temp_gen_py):
            print("Clearing COM object cache...")
            shutil.rmtree(temp_gen_py, ignore_errors=True)

        gencache.EnsureDispatch('CSTStudio.Application')
        print("COM cache refreshed successfully")
    except Exception as e:
        print(f"COM cache cleanup warning: {str(e)}")

def coth(x):
    return np.cosh(x) / np.sinh(x)

def calculate_s21_te(freq_array):
    """
    Calculate S21 TE response using the same frequency array as CST simulation
    """
    # Filter specifications
    f0 = 79e9  # Center frequency in Hz
    FBW = 0.002  # Fractional bandwidth
    BW = f0 * FBW  # Absolute bandwidth in Hz

    # Convert input frequency from GHz to Hz for calculations
    f = freq_array * 1e9  # Convert GHz to Hz

    # Filter parameters
    N = 1  # Filter order
    Lr_dB = -30  # Reflection coefficient in dB
    Lar = -10 * np.log10(1 - 10**(0.1 * Lr_dB))  # Pass band ripple

    # Prototype filter design (g-values calculation)
    g = np.zeros(N + 2)
    beta = np.log(coth(Lar / 17.37))
    gamma = np.sinh(beta / (2 * N))

    g[0] = 1
    g[1] = 2 * np.sin(np.pi / (2 * N)) / gamma

    for i in range(2, N + 1):
        numerator = 4 * np.sin((2 * i - 1) * np.pi / (2 * N)) * np.sin((2 * i - 3) * np.pi / (2 * N))
        denominator = gamma**2 + (np.sin((i - 1) * np.pi / N))**2
        g[i] = (1 / g[i - 1]) * (numerator / denominator)

    if N % 2 == 0:
        g[N + 1] = (coth(beta / 4))**2
    else:
        g[N + 1] = 1

    # Coupling matrix calculation
    R = np.zeros((N + 2, N + 2))
    R[0, 1] = 1 / np.sqrt(g[0] * g[1])
    R[N, N + 1] = 1 / np.sqrt(g[N] * g[N + 1])

    for i in range(1, N):
        R[i, i + 1] = 1 / np.sqrt(g[i] * g[i + 1])

    R1 = R.T
    M_coupling = R1 + R  # Complete coupling matrix

    # External quality factors
    Qe1 = f0 / (BW * R[0, 1])
    Qen = f0 / (BW * R[N, N + 1])

    # Frequency response calculation
    U = np.eye(M_coupling.shape[0])
    U[0, 0] = 0
    U[-1, -1] = 0

    R_matrix = np.zeros_like(M_coupling)
    R_matrix[0, 0] = 1
    R_matrix[-1, -1] = 1

    S21 = np.zeros_like(f, dtype=complex)

    for i in range(len(f)):
        lam = (f0 / BW) * ((f[i] / f0) - (f0 / f[i]))
        A = lam * U - 1j * R_matrix + M_coupling
        A_inv = np.linalg.inv(A)
        S21[i] = -2j * A_inv[-1, 0]

    return np.abs(S21), f

def mirror_8_fold_odd(base):
    """Mirror 8-fold for odd-sized matrix"""
    m = base.shape[0]
    c = m // 2
    matrix = np.zeros_like(base)
    # Mirror the values across 8 symmetric directions
    for i in range(c + 1):  # include center
        for j in range(i, c + 1):
            val = base[i, j]
            # 8 symmetric positions
            coords = [
                (i, j), (j, i),
                (i, m - 1 - j), (j, m - 1 - i),
                (m - 1 - i, j), (m - 1 - j, i),
                (m - 1 - i, m - 1 - j), (m - 1 - j, m - 1 - i)
            ]
            for x, y in coords:
                matrix[x, y] = val
    return matrix

def generate_1_8th_shape_odd(m):
    """Generate 1/8th triangle for odd-sized matrix with values 0, 1, or 2"""
    if m % 2 == 0:
        raise ValueError("Matrix size must be odd.")
    c = m // 2
    shape = np.zeros((m, m), dtype=int)
    # Fill 1/8th triangle including the center row and column
    for i in range(c + 1):  # include center
        for j in range(i, c + 1):
            shape[i, j] = np.random.choice([0, 1, 2]) # Randomly choose 0, 1, or 2
    return shape

def generate_1_8th_shape_even(m):
    """Generate 1/8th triangle for even-sized matrix with values 0, 1, or 2"""
    if m % 2 != 0:
        raise ValueError("Matrix size must be even.")
    shape = np.zeros((m, m), dtype=int)
    half = m // 2
    # Fill only lower triangle (including diagonal) of upper-left quadrant
    for i in range(half):
        for j in range(i + 1):  # includes diagonal
            shape[i, j] = np.random.choice([0, 1, 2]) # Randomly choose 0, 1, or 2
    return shape

def mirror_8_fold_even(base):
    """Mirror 8-fold for even-sized matrix"""
    m = base.shape[0]
    half = m // 2
    matrix = np.zeros_like(base)
    for i in range(half):
        for j in range(i + 1):
            val = base[i, j]
            matrix[i, j] = val
            matrix[j, i] = val
            matrix[i, m - 1 - j] = val
            matrix[j, m - 1 - i] = val
            matrix[m - 1 - i, j] = val
            matrix[m - 1 - j, i] = val
            matrix[m - 1 - i, m - 1 - j] = val
            matrix[m - 1 - j, m - 1 - i] = val
    return matrix

def generate_symmetric_pattern(n):
    """Generate symmetric pattern with 8-fold symmetry"""
    if n % 2 == 1:  # Odd n
        shape_1_8 = generate_1_8th_shape_odd(n)
        full_shape = mirror_8_fold_odd(shape_1_8)
    else:  # Even n
        shape_1_8 = generate_1_8th_shape_even(n)
        full_shape = mirror_8_fold_even(shape_1_8)

    return full_shape

    
def calculate_pcr(matrix, inputparameters):
    clear_com_cache()
    print("\nRunning CST simulation...")

    frequency = float(inputparameters[0])
    bandwidth = float(inputparameters[1])
    unitcellsize = float(inputparameters[2])
    substrateThickness = float(inputparameters[3])
    nPix = int(inputparameters[4])
    substrate = inputparameters[6]

    te = np.array([0])
    freq = np.array([frequency])
    S = np.zeros((1, 4))

    x, y = np.meshgrid(np.linspace(0, unitcellsize, nPix + 1),
                       np.linspace(0, unitcellsize, nPix + 1))
    y = np.flipud(y)

    myCST = None
    try:
        projectName = "filtermetasurface"
        myCST = cpa.CST_MicrowaveStudio(context.dataFolder, projectName + ".cst")
        myCST.Solver.defineFloquetModes(nModes=2, theta=0.0, phi=0.0, forcePolar=False, polarAngle=0.0)
        myCST.Solver.setBoundaryCondition(
            xMin="unit cell", xMax="unit cell",
            yMin="unit cell", yMax="unit cell",
            zMin="expanded open", zMax="expanded open"
        )
        myCST.Build.Material.addNormalMaterial("FR4 (Lossy)", 4.3, 1.0, colour=[0.94, 0.82, 0.76], tanD=0.025)
        myCST.Build.Shape.addBrick(
            xMin=0.0, xMax=unitcellsize,
            yMin=0.0, yMax=unitcellsize,
            zMin=0.0, zMax=substrateThickness,
            name="Substrate", component="component1", material="FR4 (Lossy)"
        )

        ii = 0
        Zblock = [substrateThickness, substrateThickness]
        for i1 in range(nPix):
            for j1 in range(nPix):
                if matrix[i1, j1] > 0: # Create geometry for 1 (metal) and 2 (metal with Jerusalem cross)
                    ii += 1
                    Xblock = [x[i1, j1], x[i1, j1 + 1]]
                    Yblock = [y[i1 + 1, j1], y[i1, j1]]
                    name = f"Brick{ii}"

                    myCST.Build.Shape.addBrick(
                        xMin=float(Xblock[0]), xMax=float(Xblock[1]),
                        yMin=float(Yblock[0]), yMax=float(Yblock[1]),
                        zMin=float(Zblock[0]), zMax=float(Zblock[1]),
                        name=name, component="component1", material="PEC"
                    )

                    if matrix[i1, j1] == 2: # Jerusalem cross slot
                        # Calculate dimensions for Jerusalem cross
                        pixel_width = Xblock[1] - Xblock[0]
                        center_x = (Xblock[0] + Xblock[1]) / 2
                        center_y = (Yblock[0] + Yblock[1]) / 2
                        
                        # Create vertical slot
                        vert_slot_name = f"VertSlot{ii}"
                        myCST.Build.Shape.addBrick(
                            xMin=center_x - pixel_width/8, xMax=center_x + pixel_width/8,
                            yMin=Yblock[0], yMax=Yblock[1],
                            zMin=Zblock[0], zMax=Zblock[1] + 0.1,
                            name=vert_slot_name, component="component1", material="PEC"
                        )
                        
                        # Create horizontal slot
                        horiz_slot_name = f"HorizSlot{ii}"
                        myCST.Build.Shape.addBrick(
                            xMin=Xblock[0], xMax=Xblock[1],
                            yMin=center_y - pixel_width/8, yMax=center_y + pixel_width/8,
                            zMin=Zblock[0], zMax=Zblock[1] + 0.1,
                            name=horiz_slot_name, component="component1", material="PEC"
                        )
                        
                        # Subtract both slots from the main patch
                        myCST.Build.Boolean.subtract("component1:" + name, "component1:" + vert_slot_name)
                        myCST.Build.Boolean.subtract("component1:" + name, "component1:" + horiz_slot_name)

        # Save with unique name
        save_path = r"C:/Users/IDARE_ECE/Documents/saved_cst_projects2/"
        save_file_name = "filtermetasurface2.cst"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        myCST.Solver.setFrequencyRange(frequency - bandwidth / 2, frequency + bandwidth / 2)
        myCST.Solver.changeSolverType("HF Frequency Domain")
        myCST.saveFile(save_path, save_file_name)
        myCST.Solver.SetNumberOfResultDataSamples(501)

        print("Running solver")
        myCST.Solver.runSimulation()
        freq, SZMax1ZMax1 = myCST.Results.getSParameters(0, 0, 1, 1)
        _, SZMax2ZMax1 = myCST.Results.getSParameters(0, 0, 2, 1)
        _, SZMin1ZMax1 = myCST.Results.getSParameters(-1, 0, 1, 1)
        _, SZMin2ZMax1 = myCST.Results.getSParameters(-1, 0, 2, 1)

        denominator = (abs(SZMin1ZMax1))**2 + (abs(SZMax1ZMax1))**2
        te = ((abs(SZMin1ZMax1))**2) / denominator
        S = np.column_stack((SZMax1ZMax1, SZMax2ZMax1, SZMin1ZMax1, SZMin2ZMax1))

        print("Simulation completed successfully")
        return te, freq, S

    except Exception as e:
        print(f"Error in CST simulation: {str(e)}")
        return te, freq, S
    finally:
        if myCST is not None:
            try:
                myCST.closeFile()
                print("CST file closed")
            except Exception as e:
                print(f"Warning: Could not close CST file: {str(e)}")
        clear_com_cache()


class PSO_Cost_Function:
    """Cost function class with context for particle storage"""
    def __init__(self, inputparameters):
        self.inputparameters = inputparameters
        self.current_iteration = None
        self.particle_index = None

    def set_context(self, iteration, particle_index):
        """Set context for current particle evaluation"""
        self.current_iteration = iteration
        self.particle_index = particle_index

    def __call__(self, x_vector):
        global all_particles_data, results_storage

        nPix = self.inputparameters[4]
        mse_threshold = self.inputparameters[5]
        good_threshold = 0.00000000002

        try:
            # Convert vector to symmetric matrix
            if nPix % 2 == 1:  # Odd size
                c = nPix // 2
                base_pattern = np.zeros((nPix, nPix), dtype=int)
                idx = 0
                for i in range(c + 1):
                    for j in range(i, c + 1):
                        base_pattern[i, j] = x_vector[idx]
                        idx += 1
                full_matrix = mirror_8_fold_odd(base_pattern)
            else:  # Even size
                half = nPix // 2
                base_pattern = np.zeros((nPix, nPix), dtype=int)
                idx = 0
                for i in range(half):
                    for j in range(i + 1):
                        base_pattern[i, j] = x_vector[idx]
                        idx += 1
                full_matrix = mirror_8_fold_even(base_pattern)

            # Run simulation
            te, freq, S = calculate_pcr(full_matrix, self.inputparameters)

            # Handle simulation failure
            if len(freq) == 1 and freq[0] == self.inputparameters[0]:
                mse = np.inf
            else:
                # Calculate target response
                target_y, _ = calculate_s21_te(freq)
                mse = np.mean((te - target_y) ** 2)

            # Create particle data dictionary
            particle_data = {
                'iteration': self.current_iteration,
                'particle_index': self.particle_index,
                'position_vector': x_vector.tolist(),
                'base_pattern': base_pattern.tolist(),
                'full_matrix': full_matrix.tolist(),
                'fitness': mse,
                'te_response': te.tolist(),
                's_parameters_real': np.real(S).tolist(),
                's_parameters_imag': np.imag(S).tolist(),
                'freq_array': freq.tolist()
            }

            # Append to global storage
            all_particles_data.append(particle_data)

            # Store good solutions
            if mse <= mse_threshold:
                save_dir = save_solution(full_matrix, mse, self.inputparameters, "threshold")
                plot_solution(full_matrix, te, freq, mse, save_dir)
                results_storage['below_threshold'].append({
                    'matrix': full_matrix,
                    'mse': mse,
                    'te': te,
                    'freq': freq,
                    'save_dir': save_dir
                })
            elif mse <= good_threshold:
                save_dir = save_solution(full_matrix, mse, self.inputparameters, "good")
                plot_solution(full_matrix, te, freq, mse, save_dir)
                results_storage['good_solutions'].append({
                    'matrix': full_matrix,
                    'mse': mse,
                    'te': te,
                    'freq': freq,
                    'save_dir': save_dir
                })

            print(f"Current MSE: {mse:.6f}")
            return mse

        except Exception as e:
            print(f"Cost function error: {str(e)}")
            return np.inf

def plot_solution(matrix, te, freq, mse, save_dir=None):
    """Plot and save solution visualization"""
    plt.figure(figsize=(12, 5))

    # Plot matrix
    plt.subplot(121)
    cmap = ListedColormap(['white', 'blue', 'red']) # white for 0, blue for 1, red for 2
    plt.imshow(matrix, cmap=cmap, interpolation='none', vmin=0, vmax=2)
    plt.gca().set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    plt.grid(which='minor', color='black', linestyle='-', linewidth=2)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Pixel Configuration (MSE: {mse:.4f})")
    cbar = plt.colorbar(ticks=[0, 1, 2])
    cbar.set_ticklabels(['Empty', 'Metal Patch', 'Patch with Cross'])

    # Plot TE response
    plt.subplot(122)
    frequency = inputparameters[0]
    bandwidth = inputparameters[1]
    target_y, _ = calculate_s21_te(freq)

    plt.plot(freq, te, 'b-', label='Optimized TE')
    plt.plot(freq, target_y, 'r--', label='Target TE')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Transmission Efficiency')
    plt.title('TE Response')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "solution_plot.png"))
        # Save text file with configuration
        with open(os.path.join(save_dir, "configuration.txt"), 'w') as f:
            f.write(f"Iteration: {current_iteration if 'current_iteration' in globals() else 'N/A'}\n")
            f.write(f"MSE: {mse}\n\n")
            f.write("Pixel Configuration:\n")
            np.savetxt(f, matrix, fmt='%d')
            f.write("\n\nLegend:\n")
            f.write("0: Empty\n")
            f.write("1: Metal Patch\n")
            f.write("2: Patch with Jerusalem Cross\n")
    plt.close()

def save_solution(matrix, mse, inputparameters, solution_type="best"):
    """Save solution data and CST file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(r"C:\Users\IDARE_ECE\cst-python-api\solutions", timestamp)
    os.makedirs(save_dir, exist_ok=True)

    # Save matrix and parameters
    np.save(os.path.join(save_dir, f"{solution_type}_matrix.npy"), matrix)
    with open(os.path.join(save_dir, f"{solution_type}_parameters.txt"), 'w') as f:
        f.write(f"MSE: {mse}\n")
        f.write(f"Frequency: {inputparameters[0]} GHz\n")
        f.write(f"Bandwidth: {inputparameters[1]} GHz\n")
        f.write(f"Unit cell size: {inputparameters[2]} mm\n")
        f.write(f"Substrate thickness: {inputparameters[3]} mm\n")
        f.write(f"Pixel grid: {inputparameters[4]}×{inputparameters[4]}\n")
        f.write("\nPixel Configuration Legend:\n")
        f.write("0: Empty (no pixel)\n")
        f.write("1: Square metal patch\n")
        f.write("2: Square metal patch with Jerusalem cross slot\n")
        f.write("\nMatrix Configuration:\n")
        np.savetxt(f, matrix, fmt='%d')

    return save_dir
class ConfigurableTPSO:
    """
    An implementation of Ternary Particle Swarm Optimization (TPSO)
    based on the phasor representation of states.
    """
    def __init__(self, fun: callable, nvars: int, config: TPSOConfig):
        self.fun = fun
        self.nvars = nvars
        self.config = config

        # --- TPSO-specific parameters ---
        # Define the three state phasors
        self.state_phasors = np.array([
            np.exp(1j * np.deg2rad(-120)),  # State 0: 1∠-120°
            np.exp(1j * np.deg2rad(0)),     # State 1: 1∠0°
            np.exp(1j * np.deg2rad(120))    # State 2: 1∠120°
        ])
        # Define the normalized indices for the states
        self.state_indices = np.array([1/6, 1/2, 5/6])
        self.transformation_c = 3.0 # Constant for the transformation function

        # Initialize tracking
        self.tracking = {
            'iteration': [], 'global_best': [], 'mean_fitness': [],
            'inertia': [], 'diversity': [], 'personal_bests': [],
            'plot_data': [], 'flip_probabilities': [], 'exploration': [], 'exploitation': []
        }

        self._initialize_swarm()

    def _phasor_to_int(self, phasor_vector):
        """Converts a vector of phasors back to an integer vector (0, 1, 2)."""
        int_vector = np.zeros(phasor_vector.shape, dtype=int)
        for i, p in enumerate(self.state_phasors):
            # Find where the phasor_vector matches the state phasor
            matches = np.isclose(phasor_vector, p)
            int_vector[matches] = i
        return int_vector

    def _int_to_phasor(self, int_vector):
        """Converts an integer vector (0, 1, 2) to a vector of phasors."""
        return self.state_phasors[int_vector]

    def _calculate_diversity(self):
        """Calculates diversity based on the variance of phases."""
        if len(self.positions) <= 1:
            return 0.0
        # Get the angles of all position phasors
        angles = np.angle(self.positions)
        # Calculate the circular standard deviation
        return np.mean(np.std(angles, axis=1))

    def _initialize_swarm(self):
        n_particles = self.config.n_particles

        # 1. Initialize positions with random phasors
        random_indices = np.random.randint(0, 3, size=(n_particles, self.nvars))
        self.positions = self.state_phasors[random_indices] # Complex-valued positions

        # 2. Initialize velocities to zero complex numbers
        self.velocities = np.zeros((n_particles, self.nvars), dtype=np.complex128)

        # 3. Initialize personal best positions (phasors) and fitness values
        self.best_positions = self.positions.copy()
        self.best_fvals = np.full(n_particles, np.inf)

        self.state = {
            'iteration': 0, 'fevals': 0, 'start_time': time.time(),
            'last_improvement': 0, 'global_best_fval': np.inf,
            'global_best_position': None, # This will store phasors
            'stall_counter': 0, 'prev_best': np.inf, 'max_diversity': 0.0
        }

        print("\n--- Starting Phasor-Based Ternary PSO ---")
        print("Iter | Best f(x) | Mean f(x) | Stall | Inertia | Diversity")
        print("----------------------------------------------------------")

        self._evaluate_swarm()
        self._display_progress()

    def _evaluate_swarm(self):
        self.current_fvals = []
        for i, x_phasor in enumerate(self.positions):
            if hasattr(self.fun, 'set_context'):
                self.fun.set_context(self.state['iteration'], i)
            
            # Convert phasor position to integer vector for the cost function
            x_int = self._phasor_to_int(x_phasor)
            fitness = self.fun(x_int)
            self.current_fvals.append(fitness)

        self.current_fvals = np.array(self.current_fvals)
        self.state['fevals'] += len(self.positions)

        # Update personal best
        improved_mask = self.current_fvals < self.best_fvals
        self.best_positions[improved_mask] = self.positions[improved_mask]
        self.best_fvals[improved_mask] = self.current_fvals[improved_mask]

        # Update global best
        current_best_idx = np.argmin(self.best_fvals)
        if self.best_fvals[current_best_idx] < self.state['global_best_fval']:
            self.state['global_best_fval'] = self.best_fvals[current_best_idx]
            self.state['global_best_position'] = self.best_positions[current_best_idx]
            self.state['last_improvement'] = self.state['iteration']
            self.state['stall_counter'] = 0
            self._store_best_solution() # Store solution if it improves
        else:
            self.state['stall_counter'] += 1
            
        self._store_iteration_data(self.current_fvals, np.any(improved_mask))
        self._generate_plots()


    def _update_swarm(self):
        n_particles = self.config.n_particles
        c1 = self.config.c1
        c2 = self.config.c2

        # Generate random numbers for this update step
        e1 = np.random.rand(n_particles, self.nvars)
        e2 = np.random.rand(n_particles, self.nvars)

        # Reshape for broadcasting
        e1 = e1[..., np.newaxis]
        e2 = e2[..., np.newaxis]
        
        # Convert random numbers to complex for phasor multiplication
        e1_complex = e1.squeeze(-1)
        e2_complex = e2.squeeze(-1)

        # Get personal and global best positions (phasors)
        p_best = self.best_positions
        g_best = self.state['global_best_position']

        # Update velocity using complex arithmetic (Equation 8)
        # Note: No inertia weight (w) is mentioned in the phasor velocity update equation.
        cognitive_term = c1 * e1_complex * (p_best - self.positions)
        social_term = c2 * e2_complex * (g_best - self.positions)
        
        self.velocities = self.velocities + cognitive_term + social_term
        
        # Update positions by mapping new velocities to discrete states
        self.positions = self._map_velocity_to_position(self.velocities)


    def _map_velocity_to_position(self, velocities):
        """Maps continuous complex velocity to discrete phasor states."""
        new_positions = np.zeros_like(velocities, dtype=np.complex128)
        
        for i in range(velocities.shape[0]): # For each particle
            for j in range(velocities.shape[1]): # For each variable/pixel
                v = velocities[i, j]

                # 1. Phase Normalization (Equation 2)
                phase_angle_deg = np.angle(v, deg=True)
                v_prime = (phase_angle_deg + 180) / 360  # Normalized phase in [0, 1]
                
                # 2. Distance Calculation
                distances = np.abs(v_prime - self.state_indices)
                
                # 3. Transformation Function (Equation 3/10)
                # Add a small epsilon to avoid division by zero if distance is 0
                epsilon = 1e-9
                transformed_probs = 1 - np.exp(1 - (1 / (self.transformation_c * (distances + epsilon))))
                
                T1, T2, T3 = transformed_probs[0], transformed_probs[1], transformed_probs[2]
                
                # 4. Decision-Making Logic
                r = np.random.rand()
                
                # This logic follows the text description for multiple conditions
                t1_ok, t2_ok, t3_ok = (T1 >= r), (T2 >= r), (T3 >= r)
                
                if t1_ok and not t2_ok and not t3_ok:
                    chosen_state_idx = 0
                elif not t1_ok and t2_ok and not t3_ok:
                    chosen_state_idx = 1
                elif not t1_ok and not t2_ok and t3_ok:
                    chosen_state_idx = 2
                elif t1_ok and t2_ok and not t3_ok:
                    chosen_state_idx = np.random.choice([0, 1])
                elif t1_ok and not t2_ok and t3_ok:
                    chosen_state_idx = np.random.choice([0, 2])
                elif not t1_ok and t2_ok and t3_ok:
                    chosen_state_idx = np.random.choice([1, 2])
                else: # All other cases (all true or all false)
                    chosen_state_idx = np.random.randint(0, 3)

                new_positions[i, j] = self.state_phasors[chosen_state_idx]
                
        return new_positions

    def _store_best_solution(self):
        """Store data for the current best solution. Converts from phasor to int."""
        try:
            nPix = inputparameters[4]
            # Convert the best phasor position to an integer vector
            best_pos_int = self._phasor_to_int(self.state['global_best_position'])

            if nPix % 2 == 1:
                c = nPix // 2
                base = np.zeros((nPix, nPix), dtype=int)
                idx = 0
                for i in range(c + 1):
                    for j in range(i, c + 1):
                        base[i, j] = best_pos_int[idx]
                        idx += 1
                matrix = mirror_8_fold_odd(base)
            else:
                half = nPix // 2
                base = np.zeros((nPix, nPix), dtype=int)
                idx = 0
                for i in range(half):
                    for j in range(i + 1):
                        base[i, j] = best_pos_int[idx]
                        idx += 1
                matrix = mirror_8_fold_even(base)
            
            te, freq, _ = calculate_pcr(matrix, inputparameters)

            plot_data = {
                'iteration': self.state['iteration'], 'matrix': matrix, 'te': te,
                'freq': freq, 'mse': self.state['global_best_fval']
            }
            self.tracking['plot_data'].append(plot_data)

            results_storage['all_iterations'].append({
                'iteration': self.state['iteration'],
                'mse': self.state['global_best_fval'],
                'matrix': matrix.copy()
            })

        except Exception as e:
            print(f"Error in _store_best_solution: {str(e)}")
            
    # The following methods (_optimize, _check_termination, etc.) are primarily for
    # orchestration and do not need to be changed for the phasor implementation.
    # They are included here for completeness of the class.

    def optimize(self) -> dict:
        while not self._check_termination():
            self.state['iteration'] += 1
            self._update_swarm()
            self._evaluate_swarm()
            self._display_progress()
        self._generate_final_diversity_plots()
        # Final result must be converted from phasor to int
        final_best_int = self._phasor_to_int(self.state['global_best_position'])
        results = self._prepare_results()
        results['x'] = final_best_int
        return results

    def _check_termination(self) -> bool:
        if self.state['global_best_fval'] <= self.config.mse_threshold:
            self.exit_flag = 3
            self.exit_message = f"MSE threshold ({self.config.mse_threshold}) reached"
            return True
        if self.state['iteration'] >= self.config.max_iterations:
            self.exit_flag = 0
            self.exit_message = "Maximum iterations reached"
            return True
        if self.state['stall_counter'] >= 40:
            self.exit_flag = 1
            self.exit_message = "Stall iterations limit reached"
            return True
        return False

    def _display_progress(self):
        line = (f"{self.state['iteration']:4d} | "
                f"{self.state['global_best_fval']:9.6f} | "
                f"{np.mean(self.current_fvals):9.2f} | "
                f"{self.state['stall_counter']:5d} | "
                f"  N/A   | " # Inertia not used in this model
                f"{self.tracking['diversity'][-1]:.4f}")
        print(line)

    def _prepare_results(self) -> dict:
        return {'x': self.state['global_best_position'], 'fun': self.state['global_best_fval'],
                'nit': self.state['iteration'], 'nfev': self.state['fevals'],
                'exit_flag': self.exit_flag, 'message': self.exit_message}

    def _store_iteration_data(self, current_values: np.ndarray, improved: bool):
        current_diversity = self._calculate_diversity()
        if current_diversity > self.state['max_diversity']:
            self.state['max_diversity'] = current_diversity
        div_max = self.state['max_diversity']
        xpl = (current_diversity / div_max) * 100 if div_max > 0 else 0
        xpt = ((div_max - current_diversity) / div_max) * 100 if div_max > 0 else 0

        self.tracking['iteration'].append(self.state['iteration'])
        self.tracking['global_best'].append(self.state['global_best_fval'])
        self.tracking['mean_fitness'].append(np.mean(current_values))
        self.tracking['inertia'].append(0) # Not used
        self.tracking['diversity'].append(current_diversity)
        self.tracking['personal_bests'].append(self.best_fvals.copy())
        self.tracking['exploration'].append(xpl)
        self.tracking['exploitation'].append(xpt)
        self._save_pixel_configuration()

    def _save_pixel_configuration(self):
        try:
            nPix = inputparameters[4]
            best_pos_int = self._phasor_to_int(self.state['global_best_position'])
            if nPix % 2 == 1:
                c = nPix // 2
                base = np.zeros((nPix, nPix), dtype=int)
                idx = 0
                for i in range(c + 1):
                    for j in range(i, c + 1):
                        base[i, j] = best_pos_int[idx]; idx += 1
                matrix = mirror_8_fold_odd(base)
            else:
                half = nPix // 2
                base = np.zeros((nPix, nPix), dtype=int)
                idx = 0
                for i in range(half):
                    for j in range(i + 1):
                        base[i, j] = best_pos_int[idx]; idx += 1
                matrix = mirror_8_fold_even(base)
            
            iter_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'TPSO_Results', f'iteration_{self.state["iteration"]}')
            os.makedirs(iter_dir, exist_ok=True)
            with open(os.path.join(iter_dir, 'pixel_configuration.txt'), 'w') as f:
                f.write(f"Iteration: {self.state['iteration']}\n")
                f.write(f"MSE: {self.state['global_best_fval']}\n")
                f.write("Pixel Configuration:\n")
                np.savetxt(f, matrix, fmt='%d')
        except Exception as e:
            print(f"Error saving pixel configuration: {str(e)}")

    def _generate_plots(self): # This method does not need changes
        pass # To save space, the plotting code is omitted but should be copied from your original script

    def _generate_iteration_diversity_plot(self, iteration): # This method does not need changes
        pass # To save space, the plotting code is omitted but should be copied from your original script

    def _generate_final_diversity_plots(self): # This method does not need changes
        pass # To save space, the plotting code is omitted but should be copied from your original script

    
def save_all_particles_data(all_particles_data, inputparameters, config):
    """Save all particles data to a PKL file with specified folder structure"""
    # Create folder name
    folder_name = (
        f"{inputparameters[4]}x{inputparameters[4]}"
        f"@{inputparameters[2]}_{config.n_particles}_polins_ternary"
    )

    # Create folder
    os.makedirs(folder_name, exist_ok=True)

    # Create file path
    file_path = os.path.join(folder_name, "all_particles_data.pkl")

    # Save data
    with open(file_path, 'wb') as f:
        pickle.dump(all_particles_data, f)

    # Print information
    print("\n" + "=" * 50)
    print(f"Saved all particles data to: {os.path.abspath(file_path)}")

    if all_particles_data:
        sample = all_particles_data[0]
        print("\nData shapes:")
        print(f"Position vector: {len(sample['position_vector'])} elements")
        print(f"Base pattern: {len(sample['base_pattern'])}x{len(sample['base_pattern'][0])}")
        print(f"Full matrix: {len(sample['full_matrix'])}x{len(sample['full_matrix'][0])}")
        print(f"TE response: {len(sample['te_response'])} points")
        s_real = np.array(sample['s_parameters_real'])
        print(f"S parameters: {s_real.shape[0]} frequencies x {s_real.shape[1]} ports")
        print(f"Frequency array: {len(sample['freq_array'])} points")
    print("=" * 50 + "\n")

    return file_path

def pyoptimize_te(inputparameters):
    global results_storage, all_particles_data

    print("\nStarting configurable TPSO optimization with parameters:")
    print(f"Frequency: {inputparameters[0]} GHz")
    print(f"Bandwidth: {inputparameters[1]} GHz")
    print(f"Unit cell size: {inputparameters[2]} mm")
    print(f"Substrate thickness: {inputparameters[3]} mm")
    print(f"Pixel grid: {inputparameters[4]}×{inputparameters[4]}")
    print(f"MSE threshold: {inputparameters[5]}")

    # Calculate number of variables based on symmetry
    nPix = inputparameters[4]
    if nPix % 2 == 1:  # Odd size
        c = nPix // 2
        nvars = sum(range(c + 2))  # Triangular number for 1/8th section
    else:  # Even size
        half = nPix // 2
        nvars = sum(range(half + 1))  # Triangular number for lower triangle

    config = TPSOConfig()
    pso_cost_fn = PSO_Cost_Function(inputparameters)

    start_time = time.time()
    optimizer = ConfigurableTPSO(
        fun=pso_cost_fn,
        nvars=nvars,
        config=config
    )

    results = optimizer.optimize()

    # Save all particles data
    file_path = save_all_particles_data(all_particles_data, inputparameters, config)

    # Process results
    optimal_vector = results['x']
    optimal_mse = results['fun']

    # Convert to full symmetric matrix
    if nPix % 2 == 1:  # Odd size
        c = nPix // 2
        base = np.zeros((nPix, nPix), dtype=int)
        idx = 0
        for i in range(c + 1):
            for j in range(i, c + 1):
                base[i, j] = optimal_vector[idx]
                idx += 1
        optimal_matrix = mirror_8_fold_odd(base)
    else:  # Even size
        half = nPix // 2
        base = np.zeros((nPix, nPix), dtype=int)
        idx = 0
        for i in range(half):
            for j in range(i + 1):
                base[i, j] = optimal_vector[idx]
                idx += 1
        optimal_matrix = mirror_8_fold_even(base)

    # Save and plot best solution
    if optimal_mse <= 0.000002:
        save_dir = save_solution(optimal_matrix, optimal_mse, inputparameters, "best")
        te_opt, freq_opt, S_opt = calculate_pcr(optimal_matrix, inputparameters)
        plot_solution(optimal_matrix, te_opt, freq_opt, optimal_mse, save_dir)

        results_storage['best_solution'] = {
            'matrix': optimal_matrix,
            'mse': optimal_mse,
            'te': te_opt,
            'freq': freq_opt,
            'save_dir': save_dir
        }

    print("\nOptimization results:")
    print(f"Optimal MSE: {optimal_mse}")
    print(f"Exit condition: {results['message']}")
    print(f"Function evaluations: {results['nfev']}")
    print(f"Iterations: {results['nit']}")
    print(f"Optimization time: {time.time() - start_time:.2f} seconds")

    # Print summary of found solutions
    print("\nSolution summary:")
    print(f"Below MSE threshold ({inputparameters[5]}): {len(results_storage['below_threshold'])}")
    print(f"Good solutions (≤ 0.00000000002): {len(results_storage['good_solutions'])}")

    return results_storage

if __name__ == "__main__":
    results = pyoptimize_te(inputparameters)
    print("\nOptimization completed. Results stored in:")
    for solution_type in ['below_threshold', 'good_solutions', 'best_solution']:
        if results[solution_type]:
            if solution_type == 'best_solution':
                print(f"- Best solution: {results[solution_type]['save_dir']}")
            else:
                print(f"- {len(results[solution_type])} {solution_type.replace('_', ' ')}")