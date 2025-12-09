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
import numpy as np
import itertools
import pickle
from datetime import datetime
from multiprocessing import Pool, cpu_count
import os
# Input parameters
inputparameters = [
    2,      # center frequency (GHz)
    20,      # bandwidth (GHz)
    5,       # dimension of unit cell (mm)
    0.8,     # width of pixel (mm)
    8,       # number of pixels (npix)
    0.0000001, # target mean squared error (MSE)
    0        # substrate type index
]

# Global storage for all configurations
all_configurations = []
def coth(x):
    return np.cosh(x) / np.sinh(x)

def clear_com_cache():
    """Clear COM object cache to prevent issues with CST"""
    try:
        temp_gen_py = os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Temp', 'gen_py')
        if os.path.exists(temp_gen_py):
            shutil.rmtree(temp_gen_py, ignore_errors=True)
        gencache.EnsureDispatch('CSTStudio.Application')
    except Exception as e:
        print(f"COM cache cleanup warning: {str(e)}")

def get_base_pattern_size(nPix):
    """Calculate number of elements in the 1/8th base pattern"""
    if nPix % 2 == 1:  # Odd size
        c = nPix // 2
        return sum(range(c + 2))  # Triangular number for 1/8th section
    else:  # Even size
        half = nPix // 2
        return sum(range(half + 1))  # Triangular number for lower triangle

def generate_all_base_patterns(nPix):
    """Generate all possible base patterns for the 1/8th triangle"""
    num_elements = get_base_pattern_size(nPix)
    print(f"Generating all {2**num_elements} possible base patterns...")
    return list(itertools.product([0, 1], repeat=num_elements))

def create_full_matrix(base_pattern, nPix):
    """Create full symmetric matrix from base pattern"""
    if nPix % 2 == 1:  # Odd size
        c = nPix // 2
        base = np.zeros((nPix, nPix), dtype=int)
        idx = 0
        for i in range(c + 1):
            for j in range(i, c + 1):
                base[i, j] = base_pattern[idx]
                idx += 1
        return mirror_8_fold_odd(base)
    else:  # Even size
        half = nPix // 2
        base = np.zeros((nPix, nPix), dtype=int)
        idx = 0
        for i in range(half):
            for j in range(i + 1):
                base[i, j] = base_pattern[idx]
                idx += 1
        return mirror_8_fold_even(base)

def mirror_8_fold_odd(base):
    """Mirror 8-fold for odd-sized matrix"""
    m = base.shape[0]
    c = m // 2
    matrix = np.zeros_like(base)
    for i in range(c + 1):
        for j in range(i, c + 1):
            val = base[i, j]
            coords = [
                (i, j), (j, i),
                (i, m - 1 - j), (j, m - 1 - i),
                (m - 1 - i, j), (m - 1 - j, i),
                (m - 1 - i, m - 1 - j), (m - 1 - j, m - 1 - i)
            ]
            for x, y in coords:
                matrix[x, y] = val
    return matrix

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

def calculate_pcr(matrix, inputparameters):
    """Run CST simulation and return results"""
    clear_com_cache()
    frequency = float(inputparameters[0])
    bandwidth = float(inputparameters[1])
    unitcellsize = float(inputparameters[2])
    substrateThickness = float(inputparameters[3])
    nPix = int(inputparameters[4])

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
        
        save_path = r"C:/Users/IDARE_ECE/Documents/saved_cst_projects/"
        save_file_name = "filtermetasurface.cst"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
             
        myCST.Solver.setFrequencyRange(frequency - bandwidth/2, frequency + bandwidth/2)
        myCST.Solver.changeSolverType("HF Frequency Domain")
        myCST.saveFile(save_path, save_file_name)
        myCST.Solver.SetNumberOfResultDataSamples(501)

        myCST.Solver.runSimulation()
        freq, SZMax1ZMax1 = myCST.Results.getSParameters(0, 0, 1, 1)
        _, SZMax2ZMax1 = myCST.Results.getSParameters(0, 0, 2, 1)
        _, SZMin1ZMax1 = myCST.Results.getSParameters(-1, 0, 1, 1)
        _, SZMin2ZMax1 = myCST.Results.getSParameters(-1, 0, 2, 1)
        
        denominator = (abs(SZMin1ZMax1))**2 + (abs(SZMax1ZMax1))**2
        te = ((abs(SZMin1ZMax1))**2) / denominator
        S = np.column_stack((SZMax1ZMax1, SZMax2ZMax1, SZMin1ZMax1, SZMin2ZMax1))
        
        return te, freq, S
        
    except Exception as e:
        print(f"Error in CST simulation: {str(e)}")
        return te, freq, S       
    finally:
        if myCST is not None:
            try:
                myCST.closeFile()
            except Exception as e:
                print(f"Warning: Could not close CST file: {str(e)}")
        clear_com_cache()

def calculate_s21_te(freq_array):
    """Calculate target S21 TE response"""
    f0 = 15e9  # Center frequency in Hz
    FBW = 0.01  # Fractional bandwidth
    BW = f0 * FBW  # Absolute bandwidth in Hz
    f = freq_array * 1e9  # Convert GHz to Hz
    
    N = 1  # Filter order
    Lr_dB = -30  # Reflection coefficient in dB
    Lar = -10 * np.log10(1 - 10**(0.1 * Lr_dB))  # Pass band ripple

    beta = np.log(coth(Lar / 17.37))
    gamma = np.sinh(beta / (2 * N))

    g = [1, 2 * np.sin(np.pi / (2 * N)) / gamma, 1]

    R = np.zeros((N + 2, N + 2))
    R[0, 1] = 1 / np.sqrt(g[0] * g[1])
    R[N, N + 1] = 1 / np.sqrt(g[N] * g[N + 1])
    R1 = R.T
    M_coupling = R1 + R

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

def calculate_mse(te, freq):
    """Calculate MSE between simulated and target TE response"""
    target_y, _ = calculate_s21_te(freq)
    return np.mean((te - target_y) ** 2)

def evaluate_configuration(base_pattern, nPix):
    """Evaluate a single configuration"""
    try:
        full_matrix = create_full_matrix(base_pattern, nPix)
        te, freq, S = calculate_pcr(full_matrix, inputparameters)
        
        if len(freq) == 1 and freq[0] == inputparameters[0]:
            mse = np.inf
        else:
            mse = calculate_mse(te, freq)
        
        config_data = {
            'base_pattern': base_pattern,
            'full_matrix': full_matrix.tolist(),
            'mse': mse,
            'te_response': te.tolist(),
            's_parameters_real': np.real(S).tolist(),
            's_parameters_imag': np.imag(S).tolist(),
            'freq_array': freq.tolist()
        }
        
        return config_data
        
    except Exception as e:
        print(f"Error evaluating configuration: {str(e)}")
        return None

def save_all_configurations(configs, inputparameters):
    """Save all configurations to a pickle file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"configs_{inputparameters[4]}x{inputparameters[4]}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    
    file_path = os.path.join(folder_name, "all_configurations.pkl")
    
    with open(file_path, 'wb') as f:
        pickle.dump({
            'input_parameters': inputparameters,
            'configurations': configs,
            'timestamp': timestamp
        }, f)
    
    print(f"\nSaved all configurations to: {os.path.abspath(file_path)}")
    print(f"Total configurations: {len(configs)}")
    
    return file_path

def main():
    nPix = inputparameters[4]
    base_patterns = generate_all_base_patterns(nPix)
    
    print(f"\nEvaluating {len(base_patterns)} configurations...")
    
    for i, pattern in enumerate(base_patterns):
        config = evaluate_configuration(pattern, nPix)
        if config:
            all_configurations.append(config)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(base_patterns):
            print(f"Progress: {i + 1}/{len(base_patterns)}")
            if config:
                print(f"Current MSE: {config['mse']:.6f}")
    
    save_path = save_all_configurations(all_configurations, inputparameters)
    print(f"\nAll data saved to: {save_path}")

if __name__ == "__main__":
    main()