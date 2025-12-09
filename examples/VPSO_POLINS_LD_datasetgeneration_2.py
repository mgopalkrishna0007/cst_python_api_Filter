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
    79,      # center frequency (GHz)
    20    ,    # bandwidth (GHz)
    2,      # dimension of unit cell (mm)
    0.8,     # width of substrate (mm)
    8,       # number of pixels (npix) 
    0.0000001,   # target mean squared error (MSE)
    0        # substrate type index
]

class BPSOConfig:
    """Configuration class for Binary PSO parameters"""
    def __init__(self):
        # PSO parameters
        self.n_particles = 100
        self.max_iterations = 20
        self.w_initial = 1.2  # Initial inertia weight
        self.w_final = 0.1   # Final inertia weight
        self.c1 = 2          # Cognitive coefficient
        self.c2 = 2     # Social coefficient
        self.v_max = 6.0       # Maximum velocity
        self.v_min = -6.0      # Minimum velocity
        self.transfer_func = 'inproved v_shape'  # Options: 'sigmoid', 'tanh', 'v_shape'
        self.mse_threshold = inputparameters[5]
        
    def get_inertia_weight(self, iteration):
        """Calculate inertia weight with linear decrease"""
        return self.w_initial - (self.w_initial - self.w_final) *(iteration / self.max_iterations)

# def transfer_function(velocity, func_type='sigmoid'):
#     """Apply transfer function to convert velocity to probability"""
#     if func_type == 'sigmoid':
#         return 1.0 / (1.0 + np.exp(-velocity))
#     elif func_type == 'tanh':
#         return (np.tanh(velocity) + 1) / 2
#     elif func_type == 'v_shape':
#         return np.abs(np.tanh(velocity))
#     else:
#         return 1.0 / (1.0 + np.exp(-velocity))  # Default to sigmoid

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
    FBW = 0.01# Fractional bandwidth
    BW = f0 * FBW  # Absolute bandwidth in Hz

    # Convert input frequency from GHz to Hz for calculations
    f = freq_array * 1e9  # Convert GHz to Hz
    
    # Filter parameters
    N = 2 # Filter order
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

def generate_1_8th_shape_odd(m):
    """Generate 1/8th triangle for odd-sized matrix"""
    if m % 2 == 0:
        raise ValueError("Matrix size must be odd.")
    c = m // 2
    shape = np.zeros((m, m), dtype=int)
    # Fill 1/8th triangle including the center row and column
    for i in range(c + 1):  # include center
        for j in range(i, c + 1):
            shape[i, j] = np.random.randint(0, 2)
    return shape

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

def generate_1_8th_shape_even(m):
    """Generate 1/8th triangle for even-sized matrix"""
    if m % 2 != 0:
        raise ValueError("Matrix size must be even.")
    
    shape = np.zeros((m, m), dtype=int)
    half = m // 2
    # Fill only lower triangle (including diagonal) of upper-left quadrant
    for i in range(half):
        for j in range(i + 1):  # includes diagonal
            shape[i, j] = np.random.randint(0, 2)
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



# def calculate_pcr(matrix, inputparameters, overlap_factor=1.5):  
#     # Added overlap_factor parameter (default 10% enlargement)
#     print("\nRunning CST simulation...")
    
#     frequency = float(inputparameters[0])
#     bandwidth = float(inputparameters[1])
#     unitcellsize = float(inputparameters[2])
#     substrateThickness = float(inputparameters[3])
#     nPix = int(inputparameters[4])
#     substrate = inputparameters[6] 

#     te = np.array([0])         
#     freq = np.array([frequency]) 
#     S = np.zeros((1, 4))
    
#     # Calculate original pixel size
#     original_pixel_size = unitcellsize / nPix
    
#     # Calculate enlarged pixel size
#     enlarged_pixel_size = original_pixel_size * overlap_factor
    
#     # Calculate the offset needed to center the enlarged pixels
#     offset = (enlarged_pixel_size - original_pixel_size) / 2
    
#     # Create grid with enlarged pixels
#     x, y = np.meshgrid(np.linspace(0, unitcellsize, nPix + 1),
#                        np.linspace(0, unitcellsize, nPix + 1))
#     y = np.flipud(y)

#     myCST = None
#     try:
#         projectName = "filtermetasurface"
#         myCST = cpa.CST_MicrowaveStudio(context.dataFolder, projectName + ".cst")
                
#         myCST.Solver.defineFloquetModes(nModes=2, theta=0.0, phi=0.0, forcePolar=False, polarAngle=0.0)
#         myCST.Solver.setBoundaryCondition(
#             xMin="unit cell", xMax="unit cell",
#             yMin="unit cell", yMax="unit cell",
#             zMin="expanded open", zMax="expanded open"
#         )
        
#         myCST.Build.Material.addNormalMaterial("FR4 (Lossy)", 4.3, 1.0, colour=[0.94, 0.82, 0.76])
#         myCST.Build.Shape.addBrick(
#             xMin=0.0, xMax=unitcellsize,
#             yMin=0.0, yMax=unitcellsize,
#             zMin=0.0, zMax=substrateThickness,
#             name="Substrate", component="component1", material="FR4 (Lossy)"
#         )
        
#         ii = 0
#         Zblock = [substrateThickness, substrateThickness]
#         for i1 in range(nPix):
#             for j1 in range(nPix):
#                 if matrix[i1, j1]:
#                     ii += 1
#                     # Calculate enlarged pixel coordinates with boundary checking
#                     x_min = max(0.0, x[i1, j1] - offset)
#                     x_max = min(unitcellsize, x[i1, j1 + 1] + offset)
#                     y_min = max(0.0, y[i1 + 1, j1] - offset)
#                     y_max = min(unitcellsize, y[i1, j1] + offset)
                    
#                     name = f"Brick{ii}"
                    
#                     myCST.Build.Shape.addBrick(
#                         xMin=float(x_min), xMax=float(x_max),
#                         yMin=float(y_min), yMax=float(y_max),
#                         zMin=float(Zblock[0]), zMax=float(Zblock[1]),
#                         name=name, component="component1", material="PEC"
#                     )
        
#         # Save with unique name
#         save_path = r"C:/Users/IDARE_ECE/Documents/saved_cst_projects/"
#         save_file_name = "filtermetasurface.cst"
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
             
#         myCST.Solver.setFrequencyRange(frequency - bandwidth/2, frequency + bandwidth/2)
#         myCST.Solver.changeSolverType("HF Frequency Domain")
#         myCST.saveFile(save_path, save_file_name)
#         myCST.Solver.SetNumberOfResultDataSamples(501)

#         print("Running solver")
#         myCST.Solver.runSimulation()
#         freq, SZMax1ZMax1 = myCST.Results.getSParameters(0, 0, 1, 1)
#         _, SZMax2ZMax1 = myCST.Results.getSParameters(0, 0, 2, 1)
#         _, SZMin1ZMax1 = myCST.Results.getSParameters(-1, 0, 1, 1)
#         _, SZMin2ZMax1 = myCST.Results.getSParameters(-1, 0, 2, 1)
        
#         denominator = (abs(SZMin1ZMax1))**2 + (abs(SZMax1ZMax1))**2
#         te = ((abs(SZMin1ZMax1))**2) / denominator
#         S = np.column_stack((SZMax1ZMax1, SZMax2ZMax1, SZMin1ZMax1, SZMin2ZMax1))
        
#         print("Simulation completed successfully")
#         return te, freq, S
        
#     except Exception as e:
#         print(f"Error in CST simulation: {str(e)}")
#         return te, freq, S       
#     finally:
#         if myCST is not None:
#             try:
#                 myCST.closeFile()
#                 print("CST file closed")
#             except Exception as e:
#                 print(f"Warning: Could not close CST file: {str(e)}")
#         # clear_com_cache()

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
        
        myCST.Build.Material.addNormalMaterial("FR4 (Lossy)", 4.3, 1.0, colour=[0.94, 0.82, 0.76] , tanD=0.025)
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
                if matrix[i1, j1]:
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
        
        # Save with unique name
        save_path = r"C:/Users/IDARE_ECE/Documents/saved_cst_projects2/"
        save_file_name = "filtermetasurface2.cst"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
             
        myCST.Solver.setFrequencyRange(frequency - bandwidth/2, frequency + bandwidth/2)
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
        good_threshold = 0.0000000000002
        
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
    cmap = ListedColormap(['white', 'red'])
    plt.imshow(matrix, cmap=cmap, interpolation='none')
    plt.gca().set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    plt.grid(which='minor', color='black', linestyle='-', linewidth=2)
    plt.xticks([])
    plt.yticks([])
    plt.title(f"Pixel Configuration (MSE: {mse:.4f})")

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
    
    return save_dir

class ConfigurableBPSO:
    def __init__(self, fun: callable, nvars: int, config: BPSOConfig):
        self.fun = fun
        self.nvars = nvars
        self.config = config
        
        # Initialize tracking
        self.tracking = {
            'iteration': [],
            'global_best': [],
            'mean_fitness': [],
            'inertia': [],
            'diversity': [],
            'personal_bests': [],
            'plot_data': [],
            'flip_probabilities': [],
            'exploration': [],
            'exploitation': []
        }
        
        self._initialize_swarm()
        
    def _sigmoid(self, x):
        """Sigmoid function for converting velocity to probability"""
        return transfer_function(x, self.config.transfer_func)
    
    def _calculate_diversity(self):
        """Calculate swarm diversity metric"""
        if len(self.positions) <= 1:
            return 0.0
        # Calculate standard deviation across all dimensions for each particle
        std_per_particle = np.std(self.positions, axis=1)
        # Return mean of these standard deviations
        return np.mean(std_per_particle)

    def _initialize_swarm(self):
        n_particles = self.config.n_particles
        
        # Initialize binary positions (0 or 1)
        self.positions = np.random.randint(0, 2, size=(n_particles, self.nvars))
        
        # Initialize velocities
        self.velocities = np.random.uniform(-1, 1, size=(n_particles, self.nvars))
        
        self.best_positions = self.positions.copy()
        self.best_fvals = np.full(n_particles, np.inf)
        
        self.state = {
            'iteration': 0,
            'fevals': 0,
            'start_time': time.time(),
            'last_improvement': 0,
            'global_best_fval': np.inf,
            'global_best_position': None,
            'inertia': self.config.w_initial,
            'stall_counter': 0,
            'prev_best': np.inf,
            'max_diversity': 0.0  # Track maximum diversity
        }
        
        print("\nIter | Best f(x) | Mean f(x) | Stall | Inertia | Diversity")
        print("----------------------------------------------------------")
        
        self._evaluate_swarm()
        self._display_progress()
    
    def _evaluate_swarm(self):
        self.current_fvals = []
        for i, x in enumerate(self.positions):
            # Set context for cost function
            if hasattr(self.fun, 'set_context'):
                self.fun.set_context(self.state['iteration'], i)
            fitness = self.fun(x)
            self.current_fvals.append(fitness)
            
        self.current_fvals = np.array(self.current_fvals)
        self.state['fevals'] += len(self.positions)
        
        improved = self.current_fvals < self.best_fvals
        self.best_positions[improved] = self.positions[improved]
        self.best_fvals[improved] = self.current_fvals[improved]
        
        current_best_idx = np.argmin(self.best_fvals)
        if self.best_fvals[current_best_idx] < self.state['global_best_fval']:
            self.state['global_best_fval'] = self.best_fvals[current_best_idx]
            self.state['global_best_position'] = self.best_positions[current_best_idx]
            self.state['last_improvement'] = self.state['iteration']
            self.state['stall_counter'] = 0
            self._store_best_solution()
        else:
            self.state['stall_counter'] += 1
            
        self._store_iteration_data(self.current_fvals, any(improved))
        self._generate_plots()
        
    def _store_best_solution(self):
        """Store data for the current best solution"""
        try:
            nPix = inputparameters[4]
            if nPix % 2 == 1:  # Odd size
                c = nPix // 2
                base = np.zeros((nPix, nPix), dtype=int)
                idx = 0
                for i in range(c + 1):
                    for j in range(i, c + 1):
                        base[i, j] = self.state['global_best_position'][idx]
                        idx += 1
                matrix = mirror_8_fold_odd(base)
            else:  # Even size
                half = nPix // 2
                base = np.zeros((nPix, nPix), dtype=int)
                idx = 0
                for i in range(half):
                    for j in range(i + 1):
                        base[i, j] = self.state['global_best_position'][idx]
                        idx += 1
                matrix = mirror_8_fold_even(base)
            
            te, freq, _ = calculate_pcr(matrix, inputparameters)
            
            plot_data = {
                'iteration': self.state['iteration'],
                'matrix': matrix,
                'te': te,
                'freq': freq,
                'mse': self.state['global_best_fval']
            }
            self.tracking['plot_data'].append(plot_data)
            
            results_storage['all_iterations'].append({
                'iteration': self.state['iteration'],
                'mse': self.state['global_best_fval'],
                'matrix': matrix.copy()
            })
            
        except Exception as e:
            print(f"Error in _store_best_solution: {str(e)}")

    def _update_swarm(self):
        n_particles = self.config.n_particles
        c1 = self.config.c1
        c2 = self.config.c2
        
        # Get inertia weight for current iteration
        w = self.config.get_inertia_weight(self.state['iteration'])
        
        # Update velocity using BPSO equation
        r1 = np.random.rand(n_particles, self.nvars)
        r2 = np.random.rand(n_particles, self.nvars)
        
        cognitive = c1 * r1 * (self.best_positions - self.positions)
        social = c2 * r2 * (self.state['global_best_position'] - self.positions)
        
        self.velocities = w * self.velocities + cognitive + social
        
        # Clamp velocity to prevent explosion
        self.velocities = np.clip(self.velocities, self.config.v_min, self.config.v_max)
        
        # Calculate probabilities before updating positions
        probs = 1 - np.exp(-(np.abs(self.velocities))*(1/2)) 
        avg_flip_prob = np.mean(probs)
        
        # Update positions
        r = np.random.rand(n_particles, self.nvars)
        flipmask = r < probs
        self.positions = np.where(flipmask, 1 - self.positions, self.positions)
        
        # Store flip probability for this iteration
        self.state['current_flip_prob'] = avg_flip_prob


    def optimize(self) -> dict:
        while not self._check_termination():
            self.state['iteration'] += 1
            self._update_swarm()
            self._evaluate_swarm()
            self._display_progress()
        
        # After optimization completes, generate final diversity plots
        self._generate_final_diversity_plots()
        return self._prepare_results()

    def _check_termination(self) -> bool:
        if self.state['global_best_fval'] <= self.config.mse_threshold:
            self.exit_flag = 3
            self.exit_message = f"MSE threshold ({self.config.mse_threshold}) reached"
            return True
            
        if self.state['iteration'] >= self.config.max_iterations:
            self.exit_flag = 0
            self.exit_message = "Maximum iterations reached"
            return True
            
        if self.state['stall_counter'] >= 40:  # Reduced stall limit
            self.exit_flag = 1
            self.exit_message = "Stall iterations limit reached"
            return True
            
        return False

    def _display_progress(self):
        line = (
            f"{self.state['iteration']:4d} | "
            f"{self.state['global_best_fval']:9.6f} | "
            f"{np.mean(self.current_fvals):9.2f} | "
            f"{self.state['stall_counter']:5d} | "
            f"{self.config.get_inertia_weight(self.state['iteration']):6.3f} | "
            f"{self.tracking['diversity'][-1]:.4f}"
        )
        print(line)

    def _prepare_results(self) -> dict:
        return {
            'x': self.state['global_best_position'],
            'fun': self.state['global_best_fval'],
            'nit': self.state['iteration'],
            'nfev': self.state['fevals'],
            'exit_flag': self.exit_flag,
            'message': self.exit_message
        }

    def _store_iteration_data(self, current_values: np.ndarray, improved: bool):
        """Store data for current iteration"""
        current_diversity = self._calculate_diversity()
        
        # Update maximum diversity if needed
        if current_diversity > self.state['max_diversity']:
            self.state['max_diversity'] = current_diversity
        
        # Calculate exploration and exploitation
        div_max = self.state['max_diversity']
        xpl = (current_diversity / div_max) * 100 if div_max > 0 else 0
        xpt = ((div_max - current_diversity) / div_max) * 100 if div_max > 0 else 0
        
        # Store all tracking data at once to ensure synchronization
        self.tracking['iteration'].append(self.state['iteration'])
        self.tracking['global_best'].append(self.state['global_best_fval'])
        self.tracking['mean_fitness'].append(np.mean(current_values))
        self.tracking['inertia'].append(self.config.get_inertia_weight(self.state['iteration']))
        self.tracking['diversity'].append(current_diversity)
        self.tracking['personal_bests'].append(self.best_fvals.copy())
        self.tracking['exploration'].append(xpl)
        self.tracking['exploitation'].append(xpt)
        self.tracking['flip_probabilities'].append(self.state.get('current_flip_prob', 0))
        
        # Save pixel configuration to file
        self._save_pixel_configuration()

    def _save_pixel_configuration(self):
        """Save current best pixel configuration to a text file"""
        try:
            nPix = inputparameters[4]
            if nPix % 2 == 1:  # Odd size
                c = nPix // 2
                base = np.zeros((nPix, nPix), dtype=int)
                idx = 0
                for i in range(c + 1):
                    for j in range(i, c + 1):
                        base[i, j] = self.state['global_best_position'][idx]
                        idx += 1
                matrix = mirror_8_fold_odd(base)
            else:  # Even size
                half = nPix // 2
                base = np.zeros((nPix, nPix), dtype=int)
                idx = 0
                for i in range(half):
                    for j in range(i + 1):
                        base[i, j] = self.state['global_best_position'][idx]
                        idx += 1
                matrix = mirror_8_fold_even(base)
            
            # Create directory for iteration data
            iter_dir = os.path.join(os.path.expanduser('~'), 'Desktop', 'BPSO_Results', 
                                  f'iteration_{self.state["iteration"]}')
            os.makedirs(iter_dir, exist_ok=True)
            
            # Save matrix to text file
            with open(os.path.join(iter_dir, 'pixel_configuration.txt'), 'w') as f:
                f.write(f"Iteration: {self.state['iteration']}\n")
                f.write(f"MSE: {self.state['global_best_fval']}\n")
                f.write("Pixel Configuration:\n")
                for row in matrix:
                    f.write(' '.join(map(str, row)) + '\n')
            
        except Exception as e:
            print(f"Error saving pixel configuration: {str(e)}")

    def _generate_plots(self):
        """Generate comprehensive plots for each iteration and save to desktop"""
        if not self.tracking['plot_data']:
            return
            
        current_data = self.tracking['plot_data'][-1]
        iteration = self.state['iteration']
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'BPSO_Results')
        os.makedirs(desktop_path, exist_ok=True)
        
        # Create figure with 3 subplots (MSE, Configuration, TE Response)
        plt.figure(figsize=(18, 6))
        
        # 1. Plot MSE Progress
        plt.subplot(131)
        if len(self.tracking['iteration']) > 1:
            plt.semilogy(self.tracking['iteration'], self.tracking['global_best'], 'b-', 
                        linewidth=2, label='Best MSE')
            plt.semilogy(self.tracking['iteration'], self.tracking['mean_fitness'], 'g--', 
                        linewidth=1.5, label='Mean MSE')
        else:
            plt.semilogy(self.tracking['iteration'], self.tracking['global_best'], 'bo', 
                        markersize=8, label='Best MSE')
            plt.semilogy(self.tracking['iteration'], self.tracking['mean_fitness'], 'go', 
                        markersize=6, label='Mean MSE')
        
        plt.axhline(y=self.config.mse_threshold, color='r', linestyle=':', 
                linewidth=1, label='Threshold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('MSE (log scale)', fontsize=12)
        plt.title('MSE Progress Over Iterations', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # 2. Plot Pixel Configuration
        plt.subplot(132)
        nPix = inputparameters[4]
        matrix = current_data['matrix']
        cmap = ListedColormap(['white', 'blue'])
        img = plt.imshow(matrix, cmap=cmap, interpolation='none')
        
        # Add grid and annotations
        plt.gca().set_xticks(np.arange(-0.5, nPix, 1), minor=True)
        plt.gca().set_yticks(np.arange(-0.5, nPix, 1), minor=True)
        plt.grid(which='minor', color='black', linestyle='-', linewidth=1)
        plt.colorbar(img, fraction=0.046, pad=0.04, ticks=[0, 1])
        
        title_text = (f'Best Pixel Configuration\n'
                    f'Iteration: {iteration}\n'
                    f'MSE: {current_data["mse"]:.4e}\n'
                    f'Fill Ratio: {np.mean(matrix):.2f}')
        plt.title(title_text, fontsize=12)
        
        # 3. Plot TE Response Comparison
        plt.subplot(133)
        freq = current_data['freq']
        te = current_data['te']
        
        # Calculate target response
        frequency = inputparameters[0]
        bandwidth = inputparameters[1]
        target_y, _ = calculate_s21_te(freq)
        
        plt.plot(freq, te, 'b-', linewidth=2, label='Optimized TE')
        plt.plot(freq, target_y, 'r--', linewidth=2, label='Target TE')
        
        # Mark center frequency
        plt.axvline(x=frequency, color='k', linestyle=':', linewidth=1, alpha=0.5)
        plt.text(frequency, max(te)*0.95, f'{frequency} GHz', 
                ha='center', va='top', backgroundcolor='white')
        
        plt.xlabel('Frequency (GHz)', fontsize=12)
        plt.ylabel('Transmission Efficiency', fontsize=12)
        plt.title('TE Response Comparison', fontsize=14)
        plt.legend(fontsize=10, loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Final adjustments and saving
        plt.tight_layout()
        summary_path = os.path.join(desktop_path, f'iteration_{iteration}_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate and save diversity plot for this iteration
        self._generate_iteration_diversity_plot(iteration)

    def _generate_iteration_diversity_plot(self, iteration):
        """Generate and save diversity plot for current iteration"""
        if len(self.tracking['iteration']) < 2:
            return
            
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'BPSO_Results')
        os.makedirs(desktop_path, exist_ok=True)
        
        plt.figure(figsize=(10, 6))
        
        # Plot diversity
        plt.ylim(0.0, 1.0)
        plt.plot(self.tracking['iteration'], self.tracking['diversity'], 'b-', linewidth=2, label='Diversity')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Diversity', fontsize=12)
        plt.title('Swarm Diversity Over Iterations', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        
        # Save plot
        div_path = os.path.join(desktop_path, f'iteration_{iteration}_diversity.png')
        plt.savefig(div_path, dpi=300, bbox_inches='tight')
        plt.close()


    def _generate_final_diversity_plots(self):
        """Generate final diversity, exploration/exploitation, and flip probability plots"""
        if len(self.tracking['iteration']) < 2:
            return
            
        desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'BPSO_Results')
        os.makedirs(desktop_path, exist_ok=True)
        
        # Ensure all arrays have the same length
        min_length = min(len(self.tracking['iteration']), 
                    len(self.tracking['diversity']),
                    len(self.tracking['exploration']),
                    len(self.tracking['exploitation']),
                    len(self.tracking['flip_probabilities']))
        
        iterations = self.tracking['iteration'][:min_length]
        diversity = self.tracking['diversity'][:min_length]
        exploration = self.tracking['exploration'][:min_length]
        exploitation = self.tracking['exploitation'][:min_length]
        flip_probs = self.tracking['flip_probabilities'][:min_length]
        
        # 1. Combined diversity, exploration, and exploitation plot
        plt.figure(figsize=(12, 6))
        
        # Normalize diversity for plotting
        div_max = max(diversity) if diversity else 1
        normalized_div = [d/div_max if div_max > 0 else 0 for d in diversity]
        
        plt.plot(iterations, normalized_div, 'b-', linewidth=2, label='Diversity (normalized)')
        plt.plot(iterations, [x/100 for x in exploration], 'g--', linewidth=2, label='Exploration (%)')
        plt.plot(iterations, [x/100 for x in exploitation], 'r-.', linewidth=2, label='Exploitation (%)')
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Normalized Value', fontsize=12)
        plt.title('Diversity, Exploration and Exploitation Over Iterations', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        comb_path = os.path.join(desktop_path, 'final_diversity_exploration_exploitation.png')
        plt.savefig(comb_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Flip probability plot (only if we have data)
        if len(iterations) == len(flip_probs) and len(iterations) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, flip_probs, 'm-', linewidth=2)
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Average Flip Probability', fontsize=12)
            plt.title('Average Flip Probability Over Iterations', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            flip_path = os.path.join(desktop_path, 'final_flip_probabilities.png')
            plt.savefig(flip_path, dpi=300, bbox_inches='tight')
            plt.close()

def save_all_particles_data(all_particles_data, inputparameters, config):
    """Save all particles data to a PKL file with specified folder structure"""
    # Create folder name
    folder_name = (
        f"{inputparameters[4]}x{inputparameters[4]}"
        f"@{inputparameters[2]}_{config.n_particles}_polins"
    )
    
    # Create folder
    os.makedirs(folder_name, exist_ok=True)
    
    # Create file path
    file_path = os.path.join(folder_name, "all_particles_data.pkl")
    
    # Save data
    with open(file_path, 'wb') as f:
        pickle.dump(all_particles_data, f)
    
    # Print information
    print("\n" + "="*50)
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
    print("="*50 + "\n")
    
    return file_path

def pyoptimize_te(inputparameters):
    global results_storage, all_particles_data
    
    print("\nStarting configurable BPSO optimization with parameters:")
    print(f"Frequency: {inputparameters[0]} GHz")
    print(f"Bandwidth: {inputparameters[1]} GHz")
    print(f"Unit cell size: {inputparameters[2]} mm")
    print(f"Substrate thickness: {inputparameters[3]} mm")
    print(f"Pixel grid: {inputparameters[4]}×{inputparameters[4]}")
    print(f"MSE threshold: {inputparameters[5]}")
    print(f"Transfer function: 1 - exp(-k*|v|/2)")
    
    # Calculate number of variables based on symmetry
    nPix = inputparameters[4]
    if nPix % 2 == 1:  # Odd size
        c = nPix // 2
        nvars = sum(range(c + 2))  # Triangular number for 1/8th section
    else:  # Even size
        half = nPix // 2
        nvars = sum(range(half + 1))  # Triangular number for lower triangle
    
    config = BPSOConfig()
    pso_cost_fn = PSO_Cost_Function(inputparameters)

    start_time = time.time()
    optimizer = ConfigurableBPSO(
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
    print(f"Good solutions (≤ 0.002): {len(results_storage['good_solutions'])}")
    
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