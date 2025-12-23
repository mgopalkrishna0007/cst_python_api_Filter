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
    12.5,      # center frequency (GHz)
    5,         # bandwidth (GHz)
    3,         # dimension of unit cell (mm)
    1.52,       # width of substrate (mm)
    12,         # number of pixels (npix) 
    0.0, # target mean squared error (MSE)
    0          # substrate type index
]

# Optimization frequencies (GHz)
OPT_FREQS = [11.0, 14.0]

# Cost function weights
WEIGHT_REAL = 1.0    # w_a
WEIGHT_IMAG = 10.0     # w_b

class BPSOConfig:
    """Configuration class for Binary PSO parameters"""
    def __init__(self):
        # PSO parameters
        self.n_particles = 100
        self.max_iterations = 20
        self.w_initial = 1.2  # Initial inertia weight
        self.w_final = 0.1   # Final inertia weight
        self.c1 = 1.49          # Cognitive coefficient
        self.c2 = 1.49          # Social coefficient
        self.v_max = 6.0     # Maximum velocity
        self.v_min = -6.0    # Minimum velocity
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


def calculate_pcr(matrix, inputparameters):
    """Run CST simulation and extract S11 at specific frequencies"""
    clear_com_cache()
    print("\nRunning CST simulation...")
    
    frequency = float(inputparameters[0])
    bandwidth = float(inputparameters[1])
    unitcellsize = float(inputparameters[2])
    substrateThickness = float(inputparameters[3])
    nPix = int(inputparameters[4])
    substrate = inputparameters[6] 
    
    # Initialize return values
    te = np.array([0])
    freq = np.array([frequency])
    S = np.zeros((1, 4), dtype=complex)
    s11_at_freqs = {}
    
    x, y = np.meshgrid(np.linspace(0, unitcellsize, nPix + 1),
                       np.linspace(0, unitcellsize, nPix + 1))
    y = np.flipud(y)

    myCST = None
    try:
        projectName = "filtermetasurface"
        myCST = cpa.CST_MicrowaveStudio(context.dataFolder, projectName + ".cst")
        dembed_centrefreq = float(frequency * 1e9)        
        myCST.Solver.defineFloquetModes_Zmax_deembed(
            nModes=2,
            fCenter=dembed_centrefreq,
            theta=0.0,
            phi=0.0
        )        
        myCST.Solver.setBoundaryCondition(
            xMin="unit cell", xMax="unit cell",
            yMin="unit cell", yMax="unit cell",
            zMin="expanded open", zMax="expanded open"
        )
        
        # myCST.Build.Material.addNormalMaterial("FR4 (Lossy)", 4.3, 1.0, colour=[0.94, 0.82, 0.76], tanD=0.025)
        myCST.Build.Material.addNormalMaterial(
            "Rogers RO4003C (lossy)", 
            3.38, 
            1.0, 
            colour=[0.94, 0.82, 0.76], 
            tanD=0.0027, 
            sigma=0.0, 
            tanDM=0.0, 
            sigmaM=0.0
        )        

        # myCST.Build.Shape.addBrick(
        #     xMin=0.0, xMax=unitcellsize,
        #     yMin=0.0, yMax=unitcellsize,
        #     zMin=0.0, zMax=substrateThickness,
        #     name="Substrate", component="component1", material="FR4 (Lossy)"
        # )

        myCST.Build.Shape.addBrick(
        xMin=0.0, xMax=unitcellsize,
        yMin=0.0, yMax=unitcellsize,
        zMin=0.0, zMax=substrateThickness,
        name="Substrate", component="component1", material= "Rogers RO4003C (lossy)"

        )
        myCST.Build.Shape.addBrick(
            xMin=0.0, xMax=unitcellsize,
            yMin=0.0, yMax=unitcellsize,
            zMin=0.0, zMax=0.0,
            name="groundplane", component="component2", material="PEC"
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
        myCST.Solver.SetNumberOfResultDataSamples(201)

        print("Running solver")
        myCST.Solver.runSimulation()
        freq, SZMax1ZMax1 = myCST.Results.getSParameters(0, 0, 1, 1)
        _, SZMax2ZMax1 = myCST.Results.getSParameters(0, 0, 2, 1)
        _, SZMin1ZMax1 = myCST.Results.getSParameters(-1, 0, 1, 1)
        _, SZMin2ZMax1 = myCST.Results.getSParameters(-1, 0, 2, 1)
        
        denominator = (np.abs(SZMin1ZMax1))**2 + (np.abs(SZMax1ZMax1))**2
        te = 1 - (((np.abs(SZMin1ZMax1))**2) / denominator)
        S = np.column_stack((SZMax1ZMax1, SZMax2ZMax1, SZMin1ZMax1, SZMin2ZMax1))
        
        # Extract S11 at target frequencies
        s11_at_freqs = extract_s11_at_frequencies(freq, S, OPT_FREQS)
        
        print("Simulation completed successfully")
        return te, freq, S, s11_at_freqs
        
    except Exception as e:
        print(f"Error in CST simulation: {str(e)}")
        return te, freq, S, s11_at_freqs       
    finally:
        if myCST is not None:
            try:
                myCST.closeFile()
                print("CST file closed")
            except Exception as e:
                print(f"Warning: Could not close CST file: {str(e)}")
        clear_com_cache()



def extract_s11_at_frequencies(freq_array, S_array, target_freqs):
    """Extract S11 at specific frequencies"""
    s11_values = {}
    
    for target_freq in target_freqs:
        # Find the closest frequency point
        idx = np.argmin(np.abs(freq_array - target_freq))
        closest_freq = freq_array[idx]
        
        # Get S11 value at this frequency
        S11_complex = S_array[idx, 0]  # SZMax1ZMax1 is the first column
        
        # Extract real and imaginary parts
        a_val = np.real(S11_complex)
        b_val = np.imag(S11_complex)
        
        # Calculate magnitude and phase
        magnitude = np.abs(S11_complex)
        phase_rad = np.angle(S11_complex)
        phase_deg = np.degrees(phase_rad)
        
        s11_values[target_freq] = {
            'real': a_val,
            'imag': b_val,
            'magnitude': magnitude,
            'phase_deg': phase_deg,
            'frequency': closest_freq,
            'S11_complex': S11_complex
        }
    
    return s11_values


def calculate_cost_from_s11(s11_values):
    """Calculate cost using magnitude and phase of S11 at target frequencies"""
    total_cost = 0.0
    
    for freq in OPT_FREQS:
        if freq in s11_values:
            mag = s11_values[freq]['magnitude']
            phase_deg = s11_values[freq]['phase_deg']
            
            # Cost terms
            # |S11| → 1
            mag_error = WEIGHT_REAL * (mag - 1.0) ** 2
            
            # phase → 0 degrees
            phase_error = WEIGHT_IMAG * (phase_deg) ** 2
            
            total_cost += mag_error + phase_error
    
    return total_cost


# def calculate_cost_from_s11(s11_values):
#     """Calculate cost using real and imaginary parts at target frequencies"""
#     total_cost = 0.0
    
#     for freq in OPT_FREQS:
#         if freq in s11_values:
#             a_val = s11_values[freq]['real']
#             b_val = s11_values[freq]['imag']
            
#             # Cost function: J = ∑[w_a*(a-1)^2 + w_b*(b)^2]
#             real_error = WEIGHT_REAL * (a_val - 1.0) ** 2
#             imag_error = WEIGHT_IMAG * (b_val) ** 2
            
#             total_cost += real_error + imag_error
    
#     return total_cost


def save_iteration_best(matrix, cost, s11_values, freq_array, S_array, iteration, is_global_best=False):
    """Save the GLOBAL BEST configuration up to this iteration to Desktop/BPSO_Results"""
    # Create BPSO_Results directory on Desktop
    desktop_path = os.path.join("C:/", "Users", "IDARE_ECE", "Desktop", "BPSO_Results")
    os.makedirs(desktop_path, exist_ok=True)
    
    # Create iteration subdirectory
    iter_dir = os.path.join(desktop_path, f"iteration_{iteration:03d}")
    os.makedirs(iter_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Save JPEG with all required plots
    fig = plt.figure(figsize=(18, 10))
    
    # Plot 1: Pixel Configuration
    plt.subplot(2, 3, 1)
    cmap = ListedColormap(['white', 'blue'])
    plt.imshow(matrix, cmap=cmap, interpolation='none')
    plt.gca().set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    plt.gca().set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    plt.grid(which='minor', color='black', linestyle='-', linewidth=2)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'Pixel Configuration\nIteration: {iteration}\nCost: {cost:.6f}')
    if is_global_best:
        plt.text(0.5, -0.1, '*** GLOBAL BEST SO FAR ***', transform=plt.gca().transAxes, 
                ha='center', fontsize=12, fontweight='bold', color='red')
    
    # Plot 2: |S11| vs Frequency
    plt.subplot(2, 3, 2)
    if len(freq_array) > 0:
        # Extract S11 magnitude from S-parameters
        S11_mag = np.abs(S_array[:, 0])
        plt.plot(freq_array, S11_mag, 'b-', linewidth=2, label='|S11|')
        
        # Mark and annotate target frequencies
        for opt_freq in OPT_FREQS:
            if opt_freq in s11_values:
                s11_data = s11_values[opt_freq]
                plt.plot(opt_freq, s11_data['magnitude'], 'ro', markersize=10)
                plt.text(opt_freq, s11_data['magnitude'], 
                        f"  {opt_freq} GHz\n  |S11|={s11_data['magnitude']:.3f}",
                        verticalalignment='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('|S11|')
        plt.title('Reflection Magnitude')
        plt.grid(True)
        plt.ylim(0, 1.2)
        plt.legend()
    
    # Plot 3: Unwrapped Phase vs Frequency
    plt.subplot(2, 3, 3)
    if len(freq_array) > 0:
        # Calculate phase from S-parameters
        S11_complex = S_array[:, 0]
        phase_rad = np.unwrap(np.angle(S11_complex))
        phase_deg = np.degrees(phase_rad)
        plt.plot(freq_array, phase_deg, 'g-', linewidth=2, label='Phase')
        
        # Mark and annotate target frequencies
        for opt_freq in OPT_FREQS:
            if opt_freq in s11_values:
                s11_data = s11_values[opt_freq]
                plt.plot(opt_freq, s11_data['phase_deg'], 'ro', markersize=10)
                plt.text(opt_freq, s11_data['phase_deg'], 
                        f"  {opt_freq} GHz\n  phase={s11_data['phase_deg']:.1f}°",
                        verticalalignment='bottom', fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Phase (degrees)')
        plt.title('Unwrapped Reflection Phase')
        plt.grid(True)
        plt.ylim(-180, 180)  # Fixed y-limits
        plt.legend()
    
    # Plot 4: Real vs Imaginary at target frequencies
    plt.subplot(2, 3, 4)
    real_vals = []
    imag_vals = []
    freq_labels = []
    
    for opt_freq in OPT_FREQS:
        if opt_freq in s11_values:
            s11_data = s11_values[opt_freq]
            real_vals.append(s11_data['real'])
            imag_vals.append(s11_data['imag'])
            freq_labels.append(f"{opt_freq} GHz")
    
    if real_vals and imag_vals:
        x_pos = np.arange(len(freq_labels))
        width = 0.35
        
        plt.bar(x_pos - width/2, real_vals, width, label='Real (a)', color='b', alpha=0.7)
        plt.bar(x_pos + width/2, imag_vals, width, label='Imag (b)', color='r', alpha=0.7)
        
        plt.xlabel('Frequency')
        plt.ylabel('S11 Value')
        plt.title('Real & Imaginary at Target Frequencies')
        plt.xticks(x_pos, freq_labels)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add target lines
        plt.axhline(y=1.0, color='b', linestyle='--', alpha=0.5, linewidth=2)
        plt.axhline(y=0.0, color='r', linestyle='--', alpha=0.5, linewidth=2)
        
        # Add value annotations
        for i, (real, imag) in enumerate(zip(real_vals, imag_vals)):
            plt.text(i - width/2, real, f'{real:.3f}', ha='center', va='bottom', fontsize=8)
            plt.text(i + width/2, imag, f'{imag:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: Cost breakdown
    plt.subplot(2, 3, 5)
    cost_breakdown = []
    freq_labels_cost = []
    
    for opt_freq in OPT_FREQS:
        if opt_freq in s11_values:
            s11_data = s11_values[opt_freq]
            real_error = WEIGHT_REAL * (s11_data['real'] - 1.0) ** 2
            imag_error = WEIGHT_IMAG * (s11_data['imag']) ** 2
            cost_breakdown.append([real_error, imag_error])
            freq_labels_cost.append(f"{opt_freq} GHz")
    
    if cost_breakdown:
        cost_breakdown = np.array(cost_breakdown)
        x_pos = np.arange(len(freq_labels_cost))
        
        plt.bar(x_pos - width/2, cost_breakdown[:, 0], width, 
                label=f'Real Error ×{WEIGHT_REAL}', color='darkblue', alpha=0.7)
        plt.bar(x_pos + width/2, cost_breakdown[:, 1], width, 
                label=f'Imag Error ×{WEIGHT_IMAG}', color='darkred', alpha=0.7)
        
        plt.xlabel('Frequency')
        plt.ylabel('Cost Contribution')
        plt.title('Cost Function Breakdown')
        plt.xticks(x_pos, freq_labels_cost)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 6: Complex plane representation
    plt.subplot(2, 3, 6, projection='polar')
    if s11_values:
        angles = []
        magnitudes = []
        freq_labels_polar = []
        
        for opt_freq in OPT_FREQS:
            if opt_freq in s11_values:
                s11_data = s11_values[opt_freq]
                angle_rad = np.radians(s11_data['phase_deg'])
                angles.append(angle_rad)
                magnitudes.append(s11_data['magnitude'])
                freq_labels_polar.append(f"{opt_freq} GHz")
        
        if angles and magnitudes:
            colors = ['red', 'blue', 'green', 'orange']
            for i, (angle, mag, label) in enumerate(zip(angles, magnitudes, freq_labels_polar)):
                plt.plot(angle, mag, 'o', markersize=10, color=colors[i % len(colors)], label=label)
                plt.text(angle, mag + 0.05, f'{mag:.2f} phase={np.degrees(angle):.0f}°', 
                        ha='center', fontsize=9)
            
            # Draw unit circle (target)
            circle_theta = np.linspace(0, 2*np.pi, 100)
            circle_r = np.ones(100)
            plt.plot(circle_theta, circle_r, 'k--', alpha=0.5, linewidth=1)
            
            plt.title('Complex Plane Representation\n(Ideal: 1 phase=0°)')
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.suptitle(f'BPSO Optimization - Iteration {iteration} - Global Best Cost: {cost:.6f}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save JPEG
    jpeg_filename = os.path.join(iter_dir, f"iteration_{iteration:03d}_global_best.jpg")
    plt.savefig(jpeg_filename, dpi=300, format='jpg', bbox_inches='tight')
    plt.close()
    
    print(f"  Saved GLOBAL BEST JPEG: {jpeg_filename}")
    
    # 2. Save TXT file with all data
    txt_filename = os.path.join(iter_dir, f"iteration_{iteration:03d}_global_best.txt")
    
    # Use UTF-8 encoding explicitly
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(f"BPSO Optimization Results - Iteration {iteration}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Cost: {cost} (GLOBAL BEST UP TO ITERATION {iteration})\n")
        f.write(f"Optimization frequencies: {OPT_FREQS} GHz\n\n")
        
        f.write("Binary Pixel Matrix:\n")
        f.write("-" * 30 + "\n")
        for row in matrix:
            f.write(' '.join(map(str, row)) + '\n')
        f.write("\n")
        
        f.write("S11 Values at Target Frequencies:\n")
        f.write("-" * 40 + "\n")
        for opt_freq in OPT_FREQS:
            if opt_freq in s11_values:
                s11_data = s11_values[opt_freq]
                f.write(f"\n{opt_freq} GHz:\n")
                f.write(f"  Real (a): {s11_data['real']:.6f}\n")
                f.write(f"  Imag (b): {s11_data['imag']:.6f}\n")
                f.write(f"  Magnitude |S11|: {s11_data['magnitude']:.6f}\n")
                f.write(f"  Phase angle: {s11_data['phase_deg']:.2f} deg\n")
                f.write(f"  Complex: {s11_data['S11_complex']:.6f}\n")
        
        f.write("\nCost Breakdown:\n")
        f.write("-" * 30 + "\n")
        total_cost_calc = 0
        for opt_freq in OPT_FREQS:
            if opt_freq in s11_values:
                s11_data = s11_values[opt_freq]
                real_error = WEIGHT_REAL * (s11_data['real'] - 1.0) ** 2
                imag_error = WEIGHT_IMAG * (s11_data['imag']) ** 2
                freq_cost = real_error + imag_error
                total_cost_calc += freq_cost
                f.write(f"\n{opt_freq} GHz:\n")
                f.write(f"  Real error: {real_error:.6f} (w_a={WEIGHT_REAL}, a={s11_data['real']:.3f})\n")
                f.write(f"  Imag error: {imag_error:.6f} (w_b={WEIGHT_IMAG}, b={s11_data['imag']:.3f})\n")
                f.write(f"  Frequency cost: {freq_cost:.6f}\n")
        f.write(f"\nTotal calculated cost: {total_cost_calc:.6f}\n")
        
        f.write("\nFull Frequency Response Data:\n")
        f.write("-" * 40 + "\n")
        f.write("Frequency(GHz)  |S11|      Phase(deg)     Real        Imag\n")
        f.write("-" * 70 + "\n")
        for i in range((len(freq_array))):
            S11 = S_array[i, 0]
            f.write(f"{freq_array[i]:12.3f}  {np.abs(S11):9.6f}  {np.degrees(np.angle(S11)):11.2f}  "
                   f"{np.real(S11):9.6f}  {np.imag(S11):9.6f}\n")

    
    print(f"  Saved TXT: {txt_filename}")
    
    return iter_dir


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
            te, freq, S, s11_values = calculate_pcr(full_matrix, self.inputparameters)
            
            # Handle simulation failure
            if len(freq) == 1 and freq[0] == self.inputparameters[0]:
                cost = np.inf
            else:
                # Calculate cost from S11 values
                cost = calculate_cost_from_s11(s11_values)
            
            # Create particle data dictionary
            particle_data = {
                'iteration': self.current_iteration,
                'particle_index': self.particle_index,
                'position_vector': x_vector.tolist(),
                'base_pattern': base_pattern.tolist(),
                'full_matrix': full_matrix.tolist(),
                'fitness': cost,
                'te_response': te.tolist(),
                's_parameters': S.tolist(),
                'freq_array': freq.tolist(),
                's11_values': s11_values
            }
            
            # Append to global storage for final pickle
            all_particles_data.append(particle_data)
            
            # Store good solutions based on cost threshold
            if cost <= mse_threshold:
                results_storage['below_threshold'].append({
                    'matrix': full_matrix,
                    'cost': cost,
                    'te': te,
                    'freq': freq,
                    's11_values': s11_values
                })
            
            print(f"  Particle {self.particle_index}: Cost = {cost:.6f}")
            return cost
            
        except Exception as e:
            print(f"Cost function error: {str(e)}")
            return np.inf


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
            'flip_probabilities': [],
            'exploration': [],
            'exploitation': []
        }
        
        self._initialize_swarm()
        
    def _sigmoid(self, x):
        """Sigmoid function for converting velocity to probability"""
        return 1 / (1 + np.exp(-x))
    
    def _calculate_diversity(self):
        """Calculate swarm diversity metric"""
        if len(self.positions) <= 1:
            return 0.0
        std_per_particle = np.std(self.positions, axis=1)
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
            'global_best_matrix': None,
            'global_best_s11': None,
            'global_best_freq': None,
            'global_best_S': None,
            'has_global_best': False,  # Add this flag
            'inertia': self.config.w_initial,
            'stall_counter': 0,
            'prev_best': np.inf,
            'max_diversity': 0.0
        }
        
        print("\nIter | Best Cost  | Mean Cost  | Stall | Inertia | Diversity")
        print("-" * 65)
        
        self._evaluate_swarm()
        self._display_progress()
    
    def _evaluate_swarm(self):
        self.current_fvals = []
        print(f"\nIteration {self.state['iteration']}: Evaluating {len(self.positions)} particles")
        
        for i, x in enumerate(self.positions):
            if hasattr(self.fun, 'set_context'):
                self.fun.set_context(self.state['iteration'], i)
            fitness = self.fun(x)
            self.current_fvals.append(fitness)
            
        self.current_fvals = np.array(self.current_fvals)
        self.state['fevals'] += len(self.positions)
        
        improved = self.current_fvals < self.best_fvals
        self.best_positions[improved] = self.positions[improved]
        self.best_fvals[improved] = self.current_fvals[improved]
        
        # Find current best particle
        current_best_idx = np.argmin(self.best_fvals)
        
        # Update global best if improved
        if self.best_fvals[current_best_idx] < self.state['global_best_fval']:
            print(f"  *** New global best found: {self.best_fvals[current_best_idx]:.6f} ***")
            self.state['global_best_fval'] = self.best_fvals[current_best_idx]
            self.state['global_best_position'] = self.best_positions[current_best_idx].copy()
            self.state['last_improvement'] = self.state['iteration']
            self.state['stall_counter'] = 0
            
            # Store the actual best particle data
            best_particle = None
            for particle in all_particles_data[-len(self.positions):]:
                if (particle['iteration'] == self.state['iteration'] and 
                    particle['particle_index'] == current_best_idx):
                    best_particle = particle
                    break
            
            if best_particle:
                self.state['global_best_matrix'] = np.array(best_particle['full_matrix'])
                self.state['global_best_s11'] = best_particle['s11_values']
                self.state['global_best_freq'] = np.array(best_particle['freq_array'])
                self.state['global_best_S'] = np.array(best_particle['s_parameters'])
                
                # Save the GLOBAL BEST configuration for this iteration
                save_iteration_best(
                    self.state['global_best_matrix'],
                    self.state['global_best_fval'],
                    self.state['global_best_s11'],
                    self.state['global_best_freq'],
                    self.state['global_best_S'],
                    self.state['iteration'],
                    is_global_best=True
                )
        else:
            self.state['stall_counter'] += 1
            
            # Save the SAME GLOBAL BEST configuration for this iteration (no improvement)
            if self.state['global_best_matrix'] is not None:
                save_iteration_best(
                    self.state['global_best_matrix'],
                    self.state['global_best_fval'],
                    self.state['global_best_s11'],
                    self.state['global_best_freq'],
                    self.state['global_best_S'],
                    self.state['iteration'],
                    is_global_best=True
                )
            else:
                # If no global best yet, save the current best of this iteration
                best_in_iter_idx = np.argmin(self.current_fvals)
                if self.current_fvals[best_in_iter_idx] < np.inf:
                    # Find this particle's data
                    for particle in all_particles_data[-len(self.positions):]:
                        if (particle['iteration'] == self.state['iteration'] and 
                            particle['particle_index'] == best_in_iter_idx):
                            # Save this as the first global best
                            self.state['global_best_matrix'] = np.array(particle['full_matrix'])
                            self.state['global_best_s11'] = particle['s11_values']
                            self.state['global_best_freq'] = np.array(particle['freq_array'])
                            self.state['global_best_S'] = np.array(particle['s_parameters'])
                            self.state['global_best_fval'] = particle['fitness']
                            self.state['global_best_position'] = self.positions[best_in_iter_idx].copy()
                            
                            save_iteration_best(
                                self.state['global_best_matrix'],
                                self.state['global_best_fval'],
                                self.state['global_best_s11'],
                                self.state['global_best_freq'],
                                self.state['global_best_S'],
                                self.state['iteration'],
                                is_global_best=True
                            )
                            break
        
        self._store_iteration_data(self.current_fvals, any(improved))
        
    def _update_swarm(self):
        n_particles = self.config.n_particles
        c1 = self.config.c1
        c2 = self.config.c2
        
        w = self.config.get_inertia_weight(self.state['iteration'])
        
        r1 = np.random.rand(n_particles, self.nvars)
        r2 = np.random.rand(n_particles, self.nvars)
        
        cognitive = c1 * r1 * (self.best_positions - self.positions)
        social = c2 * r2 * (self.state['global_best_position'] - self.positions)
        
        self.velocities = w * self.velocities + cognitive + social
        
        self.velocities = np.clip(self.velocities, self.config.v_min, self.config.v_max)
        
        probs = self._sigmoid(self.velocities)
        avg_flip_prob = np.mean(probs)
        
        r = np.random.rand(n_particles, self.nvars)
        flipmask = r < probs
        self.positions = np.where(flipmask, 1 - self.positions, self.positions)
        
        self.state['current_flip_prob'] = avg_flip_prob

    def optimize(self) -> dict:
        while not self._check_termination():
            self.state['iteration'] += 1
            self._update_swarm()
            self._evaluate_swarm()
            self._display_progress()
        
        return self._prepare_results()

    def _check_termination(self) -> bool:
        if self.state['global_best_fval'] <= self.config.mse_threshold:
            self.exit_flag = 3
            self.exit_message = f"Cost threshold ({self.config.mse_threshold}) reached"
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
        line = (
            f"{self.state['iteration']:4d} | "
            f"{self.state['global_best_fval']:10.6f} | "
            f"{np.mean(self.current_fvals):10.6f} | "
            f"{self.state['stall_counter']:5d} | "
            f"{self.config.get_inertia_weight(self.state['iteration']):7.3f} | "
            f"{self._calculate_diversity():.4f}"
        )
        print(line)

    def _prepare_results(self) -> dict:
        return {
            'x': self.state['global_best_position'],
            'fun': self.state['global_best_fval'],
            'matrix': self.state['global_best_matrix'],
            's11_values': self.state['global_best_s11'],
            'nit': self.state['iteration'],
            'nfev': self.state['fevals'],
            'exit_flag': self.exit_flag,
            'message': self.exit_message
        }

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
        self.tracking['inertia'].append(self.config.get_inertia_weight(self.state['iteration']))
        self.tracking['diversity'].append(current_diversity)
        self.tracking['personal_bests'].append(self.best_fvals.copy())
        self.tracking['exploration'].append(xpl)
        self.tracking['exploitation'].append(xpt)
        self.tracking['flip_probabilities'].append(self.state.get('current_flip_prob', 0))


def save_all_particles_data(all_particles_data, inputparameters, config, results):
    """Save all particles data to a PKL file at the end of optimization"""
    folder_name = (
        f"{inputparameters[4]}x{inputparameters[4]}"
        f"@{inputparameters[2]}_{config.n_particles}_polins"
    )
    
    os.makedirs(folder_name, exist_ok=True)
    
    # Prepare comprehensive swarm data
    swarm_data = {
        'all_particles': all_particles_data,
        'optimization_results': results,
        'input_parameters': inputparameters,
        'config_parameters': {
            'n_particles': config.n_particles,
            'max_iterations': config.max_iterations,
            'w_initial': config.w_initial,
            'w_final': config.w_final,
            'c1': config.c1,
            'c2': config.c2,
            'v_max': config.v_max,
            'v_min': config.v_min,
            'mse_threshold': config.mse_threshold
        },
        'optimization_freqs': OPT_FREQS,
        'cost_weights': {'w_a': WEIGHT_REAL, 'w_b': WEIGHT_IMAG},
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    file_path = os.path.join(folder_name, "swarm_data.pckl")
    
    with open(file_path, 'wb') as f:
        pickle.dump(swarm_data, f)
    
    print("\n" + "="*60)
    print("SWARM DATA SAVED SUCCESSFULLY")
    print("="*60)
    print(f"File: {os.path.abspath(file_path)}")
    print(f"Timestamp: {swarm_data['timestamp']}")
    print(f"Total particles evaluated: {len(all_particles_data)}")
    print(f"Total iterations: {results['nit']}")
    print(f"Best cost achieved: {results['fun']:.6f}")
    print("="*60 + "\n")
    
    # Also save a JSON summary
    json_summary = {
        'best_cost': float(results['fun']),
        'total_iterations': results['nit'],
        'total_particles': len(all_particles_data),
        'best_s11_values': {},
        'exit_message': results['message']
    }
    
    if results['s11_values']:
        for freq, data in results['s11_values'].items():
            json_summary['best_s11_values'][str(freq)] = {
                'real': float(data['real']),
                'imag': float(data['imag']),
                'magnitude': float(data['magnitude']),
                'phase_deg': float(data['phase_deg'])
            }
    
    json_path = os.path.join(folder_name, "optimization_summary.json")
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    
    print(f"Summary saved as JSON: {json_path}")
    
    return file_path


def pyoptimize_te(inputparameters):
    global results_storage, all_particles_data
    
    print("\n" + "="*60)
    print("STARTING BPSO OPTIMIZATION")
    print("="*60)
    print(f"Frequency range: {inputparameters[0]} ± {inputparameters[1]/2} GHz")
    print(f"Unit cell size: {inputparameters[2]} mm")
    print(f"Substrate thickness: {inputparameters[3]} mm")
    print(f"Pixel grid: {inputparameters[4]}×{inputparameters[4]}")
    print(f"Cost threshold: {inputparameters[5]}")
    print(f"Target frequencies: {OPT_FREQS} GHz")
    print(f"Weights: w_a={WEIGHT_REAL} (real), w_b={WEIGHT_IMAG} (imag)")
    print("="*60 + "\n")
    
    # Calculate number of variables based on symmetry
    nPix = inputparameters[4]
    if nPix % 2 == 1:
        c = nPix // 2
        nvars = sum(range(c + 2))
        print(f"Odd matrix size: {nPix}x{nPix}")
        print(f"Variables with symmetry: {nvars} (from {nPix*nPix} total)")
    else:
        half = nPix // 2
        nvars = sum(range(half + 1))
        print(f"Even matrix size: {nPix}x{nPix}")
        print(f"Variables with symmetry: {nvars} (from {nPix*nPix} total)")
    
    config = BPSOConfig()
    pso_cost_fn = PSO_Cost_Function(inputparameters)

    print("\nInitializing swarm...")
    start_time = time.time()
    optimizer = ConfigurableBPSO(
        fun=pso_cost_fn,
        nvars=nvars,
        config=config
    )
    
    print("\nStarting optimization loop...")
    results = optimizer.optimize()
    
    # Save all swarm data to pickle file
    swarm_file_path = save_all_particles_data(all_particles_data, inputparameters, config, results)
    
    # Process results
    optimal_vector = results['x']
    optimal_cost = results['fun']
    optimal_matrix = results['matrix']
    optimal_s11 = results['s11_values']
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETED")
    print("="*60)
    print(f"Optimal Cost: {optimal_cost:.6f}")
    print(f"Exit condition: {results['message']}")
    print(f"Function evaluations: {results['nfev']}")
    print(f"Iterations: {results['nit']}")
    print(f"Optimization time: {time.time() - start_time:.2f} seconds")
    
    # Print final S11 values
    if optimal_s11:
        print("\nFINAL S11 VALUES AT TARGET FREQUENCIES:")
        print("-" * 50)
        for opt_freq in OPT_FREQS:
            if opt_freq in optimal_s11:
                s11_data = optimal_s11[opt_freq]
                real_diff = abs(s11_data['real'] - 1.0)
                imag_diff = abs(s11_data['imag'])
                print(f"{opt_freq} GHz:")
                print(f"  Real (a): {s11_data['real']:.6f} (error: {real_diff:.6f})")
                print(f"  Imag (b): {s11_data['imag']:.6f} (error: {imag_diff:.6f})")
                print(f"  Magnitude |S11|: {s11_data['magnitude']:.6f}")
                print(f"  Phase ∠S11: {s11_data['phase_deg']:.2f}°")
                print(f"  Target: 1.0 + j0.0 (|S11|=1.0, ∠S11=0.0°)")
    
    # Save final summary to Desktop
    desktop_path = os.path.join("C:/", "Users", "IDARE_ECE", "Desktop", "BPSO_Results")
    final_dir = os.path.join(desktop_path, "final_results")
    os.makedirs(final_dir, exist_ok=True)
    
    # Save final matrix and results
    np.savetxt(os.path.join(final_dir, "final_matrix.txt"), optimal_matrix, fmt='%d')
    
    with open(os.path.join(final_dir, "final_summary.txt"), 'w') as f:
        f.write("FINAL OPTIMIZATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Optimization completed: {datetime.now()}\n")
        f.write(f"Final cost: {optimal_cost}\n")
        f.write(f"Iterations: {results['nit']}\n")
        f.write(f"Exit message: {results['message']}\n\n")
        
        f.write("Best configuration matrix:\n")
        for row in optimal_matrix:
            f.write(' '.join(map(str, row)) + '\n')
        
        f.write("\nFinal S11 values:\n")
        for opt_freq in OPT_FREQS:
            if opt_freq in optimal_s11:
                s11_data = optimal_s11[opt_freq]
                f.write(f"\n{opt_freq} GHz:\n")
                f.write(f"  Real: {s11_data['real']:.6f}\n")
                f.write(f"  Imag: {s11_data['imag']:.6f}\n")
                f.write(f"  Magnitude: {s11_data['magnitude']:.6f}\n")
                f.write(f"  Phase: {s11_data['phase_deg']:.2f}°\n")
    
    print(f"\nFinal results saved to: {final_dir}")
    print(f"Swarm data saved to: {swarm_file_path}")
    print("="*60)
    
    results_storage['best_solution'] = {
        'matrix': optimal_matrix,
        'cost': optimal_cost,
        's11_values': optimal_s11,
        'swarm_file': swarm_file_path,
        'final_dir': final_dir
    }
    
    return results_storage


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BINARY PSO METASURFACE OPTIMIZATION")
    print("="*60)
    print("Target: Real reflection (S11 → 1 + j0) at 11 GHz and 14 GHz")
    print(f"Weights: Real priority (w_a={WEIGHT_REAL})  Imag (w_b={WEIGHT_IMAG})")
    print("="*60)
    
    results = pyoptimize_te(inputparameters)
    
    print("\nOPTIMIZATION SUMMARY:")
    print("-" * 40)
    print(f"Best cost achieved: {results['best_solution']['cost']:.6f}")
    print(f"Swarm data file: {results['best_solution']['swarm_file']}")
    print(f"Results folder: {results['best_solution']['final_dir']}")
    print(f"Iteration results: C:/Users/IDARE_ECE/Desktop/BPSO_Results/")
    print("\nOptimization complete!")