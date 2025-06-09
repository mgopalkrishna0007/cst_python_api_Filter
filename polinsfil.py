import numpy as np
import matplotlib.pyplot as plt
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

def generate_random_wedge(n):
    """Generate a random pattern for 1/8th triangle area."""
    base_pattern = np.zeros((n, n), dtype=int)
    # Fill approximately 25% of the wedge randomly
    for i in range(n):
        for j in range(i+1):  # only fill lower triangle (1/8th wedge)
            if np.random.random() < 0.25:  # 25% chance to fill each cell
                base_pattern[i, j] = 1
    return base_pattern

def rotate_pattern(pattern, k):
    """Rotate 90° k times."""
    return np.rot90(pattern, k=k)

def reflect_pattern(pattern, axis):
    """Reflect pattern across axis: 'horizontal' or 'vertical'."""
    if axis == 'horizontal':
        return np.flipud(pattern)
    elif axis == 'vertical':
        return np.fliplr(pattern)
    return pattern

def create_8fold_pattern(base):
    """Replicate base wedge using 8-fold symmetry."""
    n = base.shape[0]
    full = np.zeros((n, n), dtype=int)

    # Diagonal wedge (base)
    full += base

    # Generate other 7 parts by rotation and reflection
    for i in range(1, 8):
        if i == 1:
            part = reflect_pattern(base, 'vertical')
        elif i == 2:
            part = rotate_pattern(base, 1)
        elif i == 3:
            part = reflect_pattern(rotate_pattern(base, 1), 'horizontal')
        elif i == 4:
            part = rotate_pattern(base, 2)
        elif i == 5:
            part = reflect_pattern(rotate_pattern(base, 2), 'vertical')
        elif i == 6:
            part = rotate_pattern(base, 3)
        elif i == 7:
            part = reflect_pattern(rotate_pattern(base, 3), 'horizontal')

        full = np.maximum(full, part)  # avoid overlap

    return full

def plot_grid(pattern, box_size):
    n = pattern.shape[0]
    cell_size = box_size / n
    
    fig, ax = plt.subplots(figsize=(8, 8))
    for i in range(n):
        for j in range(n):
            if pattern[i, j]:
                rect = plt.Rectangle((j*cell_size, (n-i-1)*cell_size), 
                                    cell_size, cell_size, color='black')
                ax.add_patch(rect)

    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_xticks(np.arange(0, box_size+cell_size, cell_size))
    ax.set_yticks(np.arange(0, box_size+cell_size, cell_size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, which='both', color='lightgray', linestyle='-', linewidth=0.5)
    ax.set_aspect('equal')
    plt.title(f"8-Fold Symmetric Pattern\n{box_size}×{box_size} units with {n}×{n} grid")
    plt.show()

# --- PARAMETERS (edit these) ---
BOX_SIZE = 5# Physical size of the square (in arbitrary units)
GRID_SIZE = 10   # Number of cells along each axis (n × n grid)

# --- GENERATE AND PLOT ---
base = generate_random_wedge(GRID_SIZE)
full_pattern = create_8fold_pattern(base)
print (full_pattern)
plot_grid(full_pattern, BOX_SIZE)


# def calculate_pcr(matrix, inputparameters): 
#     # clear_com_cache()
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
        
#         myCST.Build.Material.addNormalMaterial("FR4 (Lossy)", 4.3, 1.0, colour=[0.94, 0.82, 0.76], tanD=0.025)
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
#                     Xblock = [x[i1, j1], x[i1, j1 + 1]]
#                     Yblock = [y[i1 + 1, j1], y[i1, j1]]
#                     name = f"Brick{ii}"
                    
#                     myCST.Build.Shape.addBrick(
#                         xMin=float(Xblock[0]), xMax=float(Xblock[1]),
#                         yMin=float(Yblock[0]), yMax=float(Yblock[1]),
#                         zMin=float(Zblock[0]), zMax=float(Zblock[1]),
#                         name=name, component="component1", material="PEC"
#                     )
        
#         # Save with unique name
#         save_path = r"C:/Users/User/Documents/saved_cst_projects2/"
#         save_file_name = "filtermetasurface2.cst"
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)
             
#         myCST.Solver.setFrequencyRange(frequency - bandwidth/2, frequency + bandwidth/2)
#         # myCST.Solver.changeSolverType("HF Frequency Domain")
#         # myCST.saveFile(save_path, save_file_name)
#         # myCST.Solver.SetNumberOfResultDataSamples(501)

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


# inputparameters = [
#     28,      # center frequency (GHz)
#     3,       # bandwidth (GHz)
#     10,      # dimension of unit cell (mm)
#     0.8,     # width of pixel (mm)
#     14,      # number of pixels (npix) 
#     0.001,   # target mean squared error (MSE)
#     0        # substrate type index (e.g., 0 = default/substrate A)
# ]
# calculate_pcr(full_pattern , inputparameters)