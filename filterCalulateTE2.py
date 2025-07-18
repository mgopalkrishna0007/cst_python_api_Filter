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
import pandas as pd
from pathlib import Path

    
def calculate_pcr(matrix, inputparameters): 
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
        # myCST.Build.Material.addNormalMaterial(
        #     "Rogers RO4003C (lossy)", 
        #     3.55, 
        #     1.0, 
        #     colour=[0.94, 0.82, 0.76], 
        #     tanD=0.0027, 
        #     sigma=0.0, 
        #     tanDM=0.0, 
        #     sigmaM=0.0
        # )        
        # myCST.Build.Material.addNormalMaterial("FR4 (Lossy)", 4.3, 1.0, colour=[0.94, 0.82, 0.76])
        myCST.Build.Shape.addBrick(
            xMin=0.0, xMax=unitcellsize,
            yMin=0.0, yMax=unitcellsize,
            zMin=0.0, zMax=substrateThickness,
            # name="Substrate", component="component1", material= "Rogers RO4003C (lossy)"
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
                # myCST.closeFile()
                print("CST file closed")
            except Exception as e:
                print(f"Warning: Could not close CST file: {str(e)}")
        # clear_com_cache()
        

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
#         save_path = r"C:/Users/User/Documents/saved_cst_projects/"
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
        
def coth(x):
    return np.cosh(x) / np.sinh(x)

def calculate_s21_te(freq_array):
    """
    Calculate S21 TE response using the same frequency array as CST simulation
    """
    # Filter specifications
    f0 = 79e9  # Center frequency in Hz
    FBW = 0.05  # Fractional bandwidth
    BW = f0 * FBW  # Absolute bandwidth in Hz

    # Convert input frequency from GHz to Hz for calculations
    f = freq_array * 1e9  # Convert GHz to Hz
    
    # Filter parameters
    N = 7  # Filter order
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

    return np.abs(S21)


# def save_to_csv(freq, te, target_y):
    # """
    # Save synchronized frequency and TE data to CSV
    # """
    # # Round to reasonable decimal places
    # freq_rounded = np.round(freq, 8)
    # te_rounded = np.round(te, 8)
    # target_y_rounded = np.round(target_y, 8)
    
    # # Create a dictionary with the data
    # data = {
    #     'freq_ghz': freq_rounded,
    #     'cst_te': te_rounded,
    #     'target_te': target_y_rounded
    # }
    
    # # Create DataFrame
    # df = pd.DataFrame(data)
    
    # # Get desktop path
    # desktop_path = str(Path.home() / "Desktop")
    # csv_path = os.path.join(desktop_path, "metasurface_data_synchronized2.csv")
    
    # # Save to CSV without scientific notation
    # df.to_csv(csv_path, index=False, float_format='%.8f')
    # print(f"Synchronized data saved to {csv_path}")
    # return csv_path


# Main execution OG
matrix =   np.array([
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,1,0,0,0,0,0,0,1,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,0,0,1,0,0,1,0,0,0],
    [0,0,0,1,0,0,1,0,0,0],
    [0,0,0,1,1,1,1,0,0,0],
    [0,1,0,0,0,0,0,0,1,0],
    [0,0,1,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,0,0]
])
inputparameters = [
    79,      # center frequency (GHz)
    20,       # bandwidth (GHz)
    2, # dimension of unit cell (mm)
    0.8,     # width of pixel (mm)
    10,       # number of pixels (npix) 
    0.001,   # target mean squared error (MSE)
    0        # substrate type index (e.g., 0 = default/substrate A)
]

# Run CST simulation
te, freq, S = calculate_pcr(matrix, inputparameters)

# Calculate target TE using the same frequency array from CST
target_y = calculate_s21_te(freq)

# # Print arrays for verification
# print("CST TE array:", te)
# print("CST TE length:", len(te))
# print("CST Frequency array (GHz):", freq)
# print("CST Frequency length:", len(freq))
# print("Target TE array:", target_y)
# print("Target TE length:", len(target_y))

# Verify arrays have same length
if len(te) == len(target_y) == len(freq):
    print(f"✓ All arrays have matching length: {len(freq)}")
else:
    print(f"⚠ Array length mismatch - CST: {len(te)}, Target: {len(target_y)}, Freq: {len(freq)}")

# Calculate MSE
mse = np.mean((te - target_y) ** 2)
print(f"Mean Squared Error: {mse}")

# Save synchronized data to CSV
# csv_path = save_to_csv(freq, te, target_y)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(freq, te, 'b-', linewidth=2, label='CST TE (Optimized)')
plt.plot(freq, target_y, 'r--', linewidth=2, label='Target TE (Filter)')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Transmission Efficiency')
plt.title('TE Response Comparison (Synchronized Frequencies)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.close()

print(f"Plot completed. Data saved to: {csv_path}")
