import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
from typing import Callable, Union, List, Dict, Optional
import time
import shutil
import win32com.client
from win32com.client import gencache
from scipy.interpolate import interp1d
import sys
from datetime import datetime
from typing import Callable, List, Dict, Optional
import warnings
import context
import cst_python_api as cpa
from scipy.optimize import minimize

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

inputparameters = [
    28,      # center frequency (GHz)
    4,       # bandwidth (GHz)
    15,      # dimension of unit cell (mm)
    0.8,     # width of pixel (mm)
    15,      # number of pixels (npix) 
    0.001,   # target mean squared error (MSE)
    0        # substrate type index (e.g., 0 = default/substrate A)
]

class BinaryParticleSwarmOptimizer:
    def __init__(self, fun: Callable, nvars: int, options: Dict = None):
        self.fun = fun
        self.nvars = nvars
        
        # Default options for binary PSO
        self.options = {
            'SwarmSize': min(100, 10 * nvars),
            'MaxIterations': 200 * nvars,
            'FunctionTolerance': 1e-12,
            'MaxStallIterations': 10,
            'CognitiveWeight': 1.0,
            'SocialWeight': 1.0,
            'InertiaRange': [0.4, 1.2],
            'InitialInertia': 1.2,
            'FinalInertia': 0.4,
            'LinearDecay': True,
            'MinFractionNeighbors': 0.05,
            'MutationRate': 0.15,
            'RestartAfterStall': True,
            'RestartDiversity': 0.5,
            'Display': 'iter',
            'HybridFcn': None,
            'PrintEvery': 1,
            'MSEThreshold': inputparameters[5],
            'VelocityLimit': 4.0  # Added for binary PSO
        }
        
        if options:
            self.options.update(options)
        
        # Initialize tracking attributes
        self.tracking = {
            'plot_data': [],
            'iteration': [],
            'global_best': [],
            'mean_fitness': [],
            'inertia': [],
            'diversity': [],
            'personal_bests': [],
            'improvement_flags': []
        }
        
        # Initialize other required attributes
        self.unique_configurations = set()
        self.current_mutation_rate = self.options['MutationRate']
        
        self._initialize_state()
        
    def _sigmoid(self, x):
        """Sigmoid function for converting velocity to probability"""
        return 1 / (1 + np.exp(-x))
    
    def _calculate_diversity(self):
        """Calculate swarm diversity metric"""
        if len(self.positions) <= 1:
            return 0.0
        centroid = np.mean(self.positions, axis=0)
        distances = np.linalg.norm(self.positions - centroid, axis=1)
        return np.mean(distances)

    def _initialize_state(self):
        n_particles = self.options['SwarmSize']
        
        # Initialize binary positions (0 or 1)
        if 'InitialSwarm' in self.options and len(self.options['InitialSwarm']) > 0:
            self.positions = np.clip(np.round(self.options['InitialSwarm']), 0, 1)
        else:
            self.positions = np.random.randint(0, 2, size=(n_particles, self.nvars))
        
        # Initialize velocities with smaller range for binary PSO
        vmax = self.options['VelocityLimit']
        self.velocities = np.random.uniform(-vmax, vmax, size=(n_particles, self.nvars))
        
        self.best_positions = self.positions.copy()
        self.best_fvals = np.full(n_particles, np.inf)
        
        self.state = {
            'iteration': 0,
            'fevals': 0,
            'start_time': time.time(),
            'last_improvement': 0,
            'global_best_fval': np.inf,
            'global_best_position': None,
            'inertia': self.options['InertiaRange'][1],
            'neighborhood_size': max(2, int(n_particles * self.options['MinFractionNeighbors'])),
            'stall_counter': 0,
            'prev_best': np.inf,
            'mutation_applied': False,
            'restart_applied': False
        }
        
        if self.options['Display'] == 'iter':
            print("\nIter | Best f(x) | Mean f(x) | Stall | Inertia | Mutation")
            print("--------------------------------------------------------")
        
        self._evaluate_swarm()
        self._display_progress()
    
    def _evaluate_swarm(self):
        self.current_fvals = np.array([self.fun(x) for x in self.positions])
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
            nPix = int(np.sqrt(self.nvars))
            matrix = self.state['global_best_position'].reshape((nPix, nPix))
            
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
        n_particles = self.options['SwarmSize']
        c1 = self.options['CognitiveWeight']
        c2 = self.options['SocialWeight']
        
        # Calculate linear decay from initial to final inertia over iterations
        max_iter = self.options['MaxIterations']
        progress = min(self.state['iteration'] / max_iter, 1.0)
        target_inertia = (self.options['InitialInertia'] - 
                        (self.options['InitialInertia'] - self.options['FinalInertia']) * progress)
        
        # Update inertia with linear decay
        self.state['inertia'] = np.clip(
            target_inertia,
            self.options['FinalInertia'],
            self.options['InitialInertia']
        )
        
        # Binary PSO specific updates
        neighbors = np.zeros((n_particles, self.state['neighborhood_size']), dtype=int)
        for i in range(n_particles):
            neighbors[i] = np.arange(i, i + self.state['neighborhood_size']) % n_particles
            
        neighbor_bests = self.best_positions[neighbors]
        neighbor_fvals = self.best_fvals[neighbors]
        best_neighbor_idx = np.argmin(neighbor_fvals, axis=1)
        best_neighbors = neighbor_bests[np.arange(n_particles), best_neighbor_idx]
        
        r1 = np.random.rand(n_particles, self.nvars)
        r2 = np.random.rand(n_particles, self.nvars)
        
        # Update velocities (standard PSO formula)
        cognitive = c1 * r1 * (self.best_positions - self.positions)
        social = c2 * r2 * (best_neighbors - self.positions)
        self.velocities = self.state['inertia'] * self.velocities + cognitive + social
        
        # Clip velocities to prevent explosion
        vmax = self.options['VelocityLimit']
        self.velocities = np.clip(self.velocities, -vmax, vmax)
        
        # Update positions using sigmoid transformation
        probs = self._sigmoid(self.velocities)
        new_positions = (np.random.rand(n_particles, self.nvars) < probs).astype(int)
        
        self.positions = new_positions

    def _apply_mutation(self):
        n_particles = self.options['SwarmSize']
        self.current_mutation_rate = min(0.5, self.options['MutationRate'] + 
                                       0.05 * self.state['stall_counter'])
        
        if np.random.rand() < 0.5:  # 50% chance to mutate each iteration
            mutate_idx = np.random.rand(n_particles) < self.current_mutation_rate
            n_mutate = np.sum(mutate_idx)
            
            if n_mutate > 0:
                print(f"\nApplying mutation to {n_mutate} particles (rate: {self.current_mutation_rate:.2f})")
                # For binary PSO, flip random bits
                mask = np.random.randint(0, 2, size=(n_mutate, self.nvars))
                self.positions[mutate_idx] = np.logical_xor(self.positions[mutate_idx], mask)
                self.velocities[mutate_idx] = 0
                self.state['mutation_applied'] = True
                return True
        return False

    def _apply_restart(self):
        n_particles = self.options['SwarmSize']
        elite_size = max(1, int(0.1 * n_particles))
        elite_idx = np.argsort(self.best_fvals)[:elite_size]
        
        new_positions = np.zeros_like(self.positions)
        new_positions[:elite_size] = self.best_positions[elite_idx]
        
        for i in range(elite_size, n_particles):
            base = self.best_positions[elite_idx[np.random.randint(0, elite_size)]]
            # For binary PSO, randomly flip some bits
            flip_mask = np.random.rand(self.nvars) < 0.2  # 20% chance to flip each bit
            new_positions[i] = np.logical_xor(base, flip_mask).astype(int)
        
        print("\nApplying swarm restart due to stall")
        self.positions = new_positions
        self.velocities = np.zeros_like(self.positions)
        self.state['stall_counter'] = 0
        self.state['restart_applied'] = True
        return True

    def optimize(self) -> Dict:
        while not self._check_termination():
            self.state['iteration'] += 1
            self.state['mutation_applied'] = False
            self.state['restart_applied'] = False
            
            self._update_swarm()
            
            if self.state['stall_counter'] > self.options['MaxStallIterations'] // 2:
                self._apply_mutation()
                
            if (self.options['RestartAfterStall'] and 
                self.state['stall_counter'] >= self.options['MaxStallIterations']):
                self._apply_restart()
            
            self._evaluate_swarm()
            
            if self.options['Display'] == 'iter' and \
               self.state['iteration'] % self.options['PrintEvery'] == 0:
                self._display_progress()
        
        if self.options['HybridFcn'] is not None:
            self._run_hybrid_optimization()
        
        return self._prepare_results()

    def _check_termination(self) -> bool:
        if self.state['global_best_fval'] <= self.options['MSEThreshold']:
            self.exit_flag = 3
            self.exit_message = f"MSE threshold ({self.options['MSEThreshold']}) reached"
            return True
            
        if self.state['iteration'] >= self.options['MaxIterations']:
            self.exit_flag = 0
            self.exit_message = "Maximum iterations reached"
            return True
            
        if self.state['stall_counter'] >= self.options['MaxStallIterations']:
            self.exit_flag = 1
            self.exit_message = "Stall iterations limit reached"
            return True
            
        if (self.state['iteration'] > 25 and 
            abs(self.state['global_best_fval'] - self.state['prev_best']) < self.options['FunctionTolerance']):
            self.exit_flag = 2
            self.exit_message = "Function tolerance reached"
            return True
            
        self.state['prev_best'] = self.state['global_best_fval']
        return False

    def _run_hybrid_optimization(self):
        try:
            # For binary problems, we might want to skip hybrid optimization
            # or implement a binary-compatible method
            print("Skipping hybrid optimization for binary PSO")
            
        except Exception as e:
            warnings.warn(f"Hybrid optimization failed: {str(e)}")

    def _display_progress(self):
        line = (
            f"{self.state['iteration']:4d} | "
            f"{self.state['global_best_fval']:9.6f} | "
            f"{np.mean(self.current_fvals):9.2f} | "
            f"{self.state['stall_counter']:5d} | "
            f"{self.state['inertia']:6.3f} | "
            f"{self.current_mutation_rate if hasattr(self, 'current_mutation_rate') else 0:.3f}"
        )
        
        if self.state['restart_applied']:
            line += " (R)"
        elif self.state['mutation_applied']:
            line += " (M)"
            
        print(line)

    def _prepare_results(self) -> Dict:
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
        self.tracking['iteration'].append(self.state['iteration'])
        self.tracking['global_best'].append(self.state['global_best_fval'])
        self.tracking['mean_fitness'].append(np.mean(current_values))
        self.tracking['inertia'].append(self.state['inertia'])
        self.tracking['diversity'].append(self._calculate_diversity())
        self.tracking['personal_bests'].append(self.best_fvals.copy())
        self.tracking['improvement_flags'].append(improved)

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
        
        plt.axhline(y=self.options['MSEThreshold'], color='r', linestyle=':', 
                linewidth=1, label='Threshold')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('MSE (log scale)', fontsize=12)
        plt.title('MSE Progress Over Iterations', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        
        # 2. Plot Pixel Configuration
        plt.subplot(132)
        nPix = int(np.sqrt(self.nvars))
        matrix = current_data['matrix'].reshape((nPix, nPix))
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
        target_x = np.linspace(frequency - bandwidth/2, frequency + bandwidth/2, len(freq))
        target_y, _ = calculate_s21_te()
        
        plt.plot(freq, te, 'b-', linewidth=2, label='Optimized TE')
        plt.plot(target_x, target_y, 'r--', linewidth=2, label='Target TE')
        
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
    target_x = np.linspace(frequency - bandwidth/2, frequency + bandwidth/2, len(freq))
    target_y, _ = calculate_s21_te()
    
    plt.plot(freq, te, 'b-', label='Optimized TE')
    plt.plot(target_x, target_y, 'r--', label='Target TE')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Transmission Efficiency')
    plt.title('TE Response')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "solution_plot.png"))
    plt.show()
    plt.close()

def save_solution(matrix, mse, inputparameters, solution_type="best"):
    """Save solution data and CST file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(r"C:\Users\User\cst-python-api\solutions", timestamp)
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

def calculate_s21_te():
    # Filter specifications
    f0 = 28e9  # Center frequency in Hz
    FBW = 0.03  # Fractional bandwidth
    BW = f0 * FBW  # Absolute bandwidth in Hz

    # Frequency range setup
    fmin = 26e9
    fmax = 30e9
    fstep = 1e6
    x1 = np.arange(fmin,fmax + 1e6, 1e6)
    len1 = len(x1)
    len2 = 501 # len of frequency
    interp_indices = np.linspace(0, len1 - 1, len2 - 2)
    interp_func = interp1d(np.arange(len1), x1)
    interpolated = interp_func(interp_indices)
    extended_x1 = np.concatenate(([x1[0]], interpolated, [x1[-1]]))
    f = extended_x1
    
    # Filter parameters
    N = 3  # Filter order
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
        save_path = r"C:/Users/User/Documents/saved_cst_projects/"
        save_file_name = "filtermetasurface.cst"
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

def create_pso_cost_wrapper(inputparameters):
    def pso_cost_function(x_vector):
        global results_storage
        
        nPix = inputparameters[4]
        mse_threshold = inputparameters[5]
        good_threshold = 0.002  # Additional threshold for "good" solutions
        
        try:
            matrix = x_vector.reshape((nPix, nPix))
            te, freq, S = calculate_pcr(matrix, inputparameters)
            
            if len(freq) == 1 and freq[0] == inputparameters[0]:
                return np.inf
            
            frequency = inputparameters[0]
            bandwidth = inputparameters[1]
            
            target_x = np.linspace(frequency - bandwidth/2, frequency + bandwidth/2, len(freq))
            target_y, _ = calculate_s21_te()
            
            mse = np.mean((te - target_y) ** 2)
            
            # Store and visualize good solutions
            if mse <= mse_threshold:
                print(f"\nFound solution below MSE threshold ({mse_threshold}): {mse:.6f}")
                save_dir = save_solution(matrix, mse, inputparameters, "threshold")
                plot_solution(matrix, te, freq, mse, save_dir)
                results_storage['below_threshold'].append({
                    'matrix': matrix,
                    'mse': mse,
                    'te': te,
                    'freq': freq,
                    'save_dir': save_dir
                })
            elif mse <= good_threshold:
                print(f"\nFound good solution (MSE ≤ {good_threshold}): {mse:.6f}")
                save_dir = save_solution(matrix, mse, inputparameters, "good")
                plot_solution(matrix, te, freq, mse, save_dir)
                results_storage['good_solutions'].append({
                    'matrix': matrix,
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
    
    return pso_cost_function

def pyoptimize_te(inputparameters):
    global results_storage
    
    print("\nStarting binary PSO optimization with parameters:")
    print(f"Frequency: {inputparameters[0]} GHz")
    print(f"Bandwidth: {inputparameters[1]} GHz")
    print(f"Unit cell size: {inputparameters[2]} mm")
    print(f"Substrate thickness: {inputparameters[3]} mm")
    print(f"Pixel grid: {inputparameters[4]}×{inputparameters[4]}")
    print(f"MSE threshold: {inputparameters[5]}")
    
    nvars = inputparameters[4] * inputparameters[4]

    pso_cost_fn = create_pso_cost_wrapper(inputparameters)

    options = {
        'SwarmSize': 50,
        'MaxIterations': 50,
        'CognitiveWeight': 2.0,
        'SocialWeight': 2.0,
        'InitialInertia': 1.2,
        'FinalInertia': 0.4,
        'LinearDecay': True,
        'VelocityLimit': 6.0,
        'Display': 'iter',
        'MSEThreshold': inputparameters[5],
        'PrintEvery': 1
    }

    start_time = time.time()
    optimizer = BinaryParticleSwarmOptimizer(
        fun=pso_cost_fn,
        nvars=nvars,
        options=options
    )
    
    results = optimizer.optimize()
    
    # Process results
    optimal_matrix = results['x'].reshape((inputparameters[4], inputparameters[4]))
    optimal_mse = results['fun']
    
    # Save and plot best solution
    if optimal_mse <= 0.002:  # Only if it's a good solution
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