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

    
# def calculate_pcr(matrix, inputparameters): 
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
    
#     x, y = np.meshgrid(np.linspace(0, unitcellsiz   e, nPix + 1),
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
        
#         # myCST.Build.Material.addNormalMaterial("FR4 (Lossy)", 4.3, 1.0, colour=[0.94, 0.82, 0.76], tanD=0.025)

#         myCST.Build.Material.addNormalMaterial(
#             name="LossyDielectric_265",
#             eps=265,
#             mu=1.0,
#             colour=[0.4, 0.7, 0.9],   # Choose any color you like
#             tanD=0.003                # Loss tangent as given
#         )

#         myCST.Build.Shape.addBrick(
#             xMin=0.0, xMax=unitcellsize,
#             yMin=0.0, yMax=unitcellsize,
#             zMin=0.0, zMax=substrateThickness,
#             name="Substrate", component="component1", material="LossyDielectric_265"
#         )
        
#         ii = 0
#         Zblock = [substrateThickness, substrateThickness + 0.018]
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
#                         name=name, component="component1", material="Copper (annealed)"
#                     )
        
#         # Save with unique name
#         save_path = r"C:/Users/GOPAL/Documents/saved_cst_projects/"
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
        

# Main execution
# Original matrix with ones replaced by zeroes and zeroes replaced by ones
import numpy as np



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
        
        # Add custom dielectric material
        myCST.Build.Material.addNormalMaterial(
            name="LossyDielectric_265",
            eps=265.0,
            mu=1.0,
            colour=[0.4, 0.7, 0.9],
            tanD=0.003
        )

        # Add substrate
        myCST.Build.Shape.addBrick(
            xMin=0.0, xMax=unitcellsize,
            yMin=0.0, yMax=unitcellsize,
            zMin=0.0, zMax=substrateThickness,
            name="Substrate", component="component1", material="LossyDielectric_265"
        )
        
        # Add ground metal patch (8x8 mm, thickness 0.18 mm)
        ground_patch_size = 8.0  # mm
        ground_patch_thickness = 0.18  # mm
        ground_x_center = unitcellsize/2
        ground_y_center = unitcellsize/2
        
        myCST.Build.Shape.addBrick(
            xMin=ground_x_center - ground_patch_size/2,
            xMax=ground_x_center + ground_patch_size/2,
            yMin=ground_y_center - ground_patch_size/2,
            yMax=ground_y_center + ground_patch_size/2,
            zMin=-ground_patch_thickness,  # Below the substrate
            zMax=0.0,
            name="GroundPatch", component="component1", material="Copper (annealed)"
        )
        
        # Add metal patches from matrix
        ii = 0
        Zblock = [substrateThickness, substrateThickness + 0.018]
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
                        name=name, component="component1", material="Copper (annealed)"
                    )
        
        # Save with unique name
        save_path = r"C:/Users/IDARE_ECE/Documents/saved_cst_projects/"
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

matrix = np.array([
    [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
    [1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
])

# Flip 0 <-> 1
flipped_matrix = 1 - matrix

print("Flipped Matrix (0s to 1s, 1s to 0s):\n", flipped_matrix)


# Invert the matrix (replace 1 with 0 and 0 with 1)
matrix = 1 - matrix

inputparameters = [
    7,      # center frequency (GHz)
    5,       # bandwidth (GHz)
    8,       # dimension of unit cell (mm)
    0.8,     # height of (mm)
    16,       # number of pixels (npix) - changed to 16 to match matrix size
    0.001,   # target mean squared error (MSE)
    0        # substrate type index (e.g., 0 = default/substrate A)
]

# Run CST simulation
te, freq, S = calculate_pcr(matrix, inputparameters)

# Plotting section
if len(freq) > 1 and len(S) > 1:
    # Create figure for S-parameters magnitude
    plt.figure(1, figsize=(12, 8))
    plt.plot(freq, 20*np.log10(np.abs(S[:, 0])), label='S11')
    plt.plot(freq, 20*np.log10(np.abs(S[:, 1])), label='S21')
    plt.plot(freq, 20*np.log10(np.abs(S[:, 2])), label='S31')
    plt.plot(freq, 20*np.log10(np.abs(S[:, 3])), label='S41')
    plt.title('S-Parameters Magnitude (dB)')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    
    # Create figure for real parts
    plt.figure(2, figsize=(12, 8))
    plt.plot(freq, np.real(S[:, 0]), label='Re(S11)')
    plt.plot(freq, np.real(S[:, 1]), label='Re(S21)')
    plt.plot(freq, np.real(S[:, 2]), label='Re(S31)')
    plt.plot(freq, np.real(S[:, 3]), label='Re(S41)')
    plt.title('S-Parameters Real Parts')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Real Part')
    plt.legend()
    plt.grid(True)
    
    # Create figure for imaginary parts
    plt.figure(3, figsize=(12, 8))
    plt.plot(freq, np.imag(S[:, 0]), label='Im(S11)')
    plt.plot(freq, np.imag(S[:, 1]), label='Im(S21)')
    plt.plot(freq, np.imag(S[:, 2]), label='Im(S31)')
    plt.plot(freq, np.imag(S[:, 3]), label='Im(S41)')
    plt.title('S-Parameters Imaginary Parts')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Imaginary Part')
    plt.legend()
    plt.grid(True)
    
    # Show all plots
    plt.show()
else:
    print("Not enough data to plot. Simulation may have failed.")