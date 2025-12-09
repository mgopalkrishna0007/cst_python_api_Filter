# 8x8 jerruselum cross and metallic patches 79 fhz band pass

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from datetime import datetime
import warnings
# The following imports are specific to the user's environment for CST integration.
# If you are running this code without CST, these will raise an error.
# You can comment them out for testing the plotting logic with dummy data.
try:
    import context
    import cst_python_api as cpa
    import win32com.client
    from win32com.client import gencache
except ImportError:
    print("Warning: CST-related libraries not found. The 'calculate_pcr' function will not work.")
    # Define dummy modules/classes if they don't exist, to avoid crashing the script
    class DummyCST:
        def __getattr__(self, name):
            if name == 'Solver' or name == 'Build' or name == 'Results':
                return self
            def dummy_method(*args, **kwargs):
                print(f"Dummy call to {name} with args: {args}, kwargs: {kwargs}")
                if name == 'getSParameters':
                    freq = np.linspace(75, 83, 501)
                    val = np.random.rand(501) + 1j * np.random.rand(501)
                    return freq, val
                return None
            return dummy_method

    cpa = type('cpa', (), {'CST_MicrowaveStudio': DummyCST})
    context = type('context', (), {'dataFolder': '.'})
    win32com = type('win32com', (), {'client': None})


def clear_com_cache():
    """
    Attempts to clear the win32com cache to prevent issues with CST communication.
    This is a robust version that handles cases where the cache directory might not exist
    or when running on a non-Windows system.
    """
    try:
        if gencache.is_generated_dir(gencache.GetGeneratePath()):
            print("Clearing COM cache...")
            shutil.rmtree(gencache.GetGeneratePath())
            print("COM cache cleared successfully.")
    except Exception as e:
        print(f"Info: Could not clear COM cache (this is expected on non-Windows systems): {e}")


def coth(x):
    """Calculates the hyperbolic cotangent."""
    return np.cosh(x) / np.sinh(x)

def calculate_mse(actual, desired):
    """Calculates the Mean Squared Error between two arrays."""
    return np.mean((actual - desired)**2)

def calculate_s21_te(freq_array):
    """
    Calculate the desired S21 TE response for a Chebyshev filter.
    This serves as the target or "desired" response.
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

    g[N + 1] = (coth(beta / 4))**2 if N % 2 == 0 else 1

    # Coupling matrix calculation
    R = np.zeros((N + 2, N + 2))
    R[0, 1] = 1 / np.sqrt(g[0] * g[1])
    R[N, N + 1] = 1 / np.sqrt(g[N] * g[N + 1])

    for i in range(1, N):
        R[i, i + 1] = 1 / np.sqrt(g[i] * g[i + 1])

    R1 = R.T
    M_coupling = R1 + R  # Complete coupling matrix

    # Frequency response calculation
    U = np.eye(M_coupling.shape[0])
    U[0, 0], U[-1, -1] = 0, 0
    R_matrix = np.zeros_like(M_coupling)
    R_matrix[0, 0], R_matrix[-1, -1] = 1, 1
    S21 = np.zeros_like(f, dtype=complex)

    for i in range(len(f)):
        lam = (f0 / BW) * ((f[i] / f0) - (f0 / f[i]))
        A = lam * U - 1j * R_matrix + M_coupling
        A_inv = np.linalg.inv(A)
        S21[i] = -2j * A_inv[-1, 0]

    return np.abs(S21)
    
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
        myCST.Build.Material.addNormalMaterial("FR4 (Lossy)", 4.3, 1.0, colour=[0.94, 0.82, 0.76])
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
                # myCST.closeFile()
                print("CST file closed")
            except Exception as e:
                print(f"Warning: Could not close CST file: {str(e)}")
        clear_com_cache()

def parse_matrices(matrix_string):
    """Parses a raw string of numbers into a list of numpy arrays."""
    matrices = []
    # Clean up the string and split into blocks for each matrix
    cleaned_string = matrix_string.strip().replace('and', '')
    blocks = cleaned_string.split('\n\n')
    
    for block in blocks:
        if not block.strip():
            continue
        lines = block.strip().split('\n')
        matrix_data = []
        for line in lines:
            row = [int(num) for num in line.strip().split()]
            matrix_data.append(row)
        matrices.append(np.array(matrix_data))
    return matrices


if __name__ == "__main__":
    # --- Input Parameters ---
    # NOTE: nPix (index 4) has been changed from 8 to 12 to match the dimensions of the provided matrices.
    inputparameters = [
        79,    # center frequency (GHz)
        20,    # bandwidth (GHz)
        2,     # dimension of unit cell (mm)
        0.8,   # width of pixel (mm) -> used as substrateThickness in the code
        8,  # number of pixels (nPix)
        1e-10, # target mean squared error (MSE) - not used directly in this script version
        0      # substrate type index - not used directly in this script version
    ]
    
    # --- Matrices Data ---
    # All matrices provided by the user are stored in this multi-line string.
    matrices_raw_string = """

1 0 0 0 0 0 0 1
0 2 0 2 2 0 2 0
0 0 2 1 1 2 0 0
0 2 1 2 2 1 2 0
0 2 1 2 2 1 2 0
0 0 2 1 1 2 0 0
0 2 0 2 2 0 2 0
1 0 0 0 0 0 0 1

    """
    
    # --- Main Processing Loop ---
    
    # Parse the raw string into a list of numpy arrays
    matrices_to_process = parse_matrices(matrices_raw_string)
    
    # Create a directory to save the plots
    output_dir = "pcr_plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Found {len(matrices_to_process)} matrices to process.")
    print(f"Plots will be saved to: {os.path.abspath(output_dir)}")

    # Loop through each matrix
    for i, matrix in enumerate(matrices_to_process):
        matrix_number = i + 1
        print("-" * 50)
        print(f"Processing Matrix {matrix_number}/{len(matrices_to_process)}")
        
        # 1. Get the actual TE response from the CST simulation
        # The 'S' variable is returned but not used in the plotting logic.
        actual_te, freq_array, S = calculate_pcr(matrix, inputparameters)
        
        # Check if the simulation returned valid data
        if freq_array is None or len(freq_array) <= 1:
            print(f"Skipping plot for Matrix {matrix_number} due to invalid frequency data from simulation.")
            continue
            
        # 2. Get the desired (ideal) TE response
        desired_te = calculate_s21_te(freq_array)
        
        # 3. Calculate the Mean Squared Error
        mse = calculate_mse(actual_te, desired_te)
        print(f"Matrix {matrix_number} - MSE: {mse:.6f}")
        
        # 4. Create and save the plot
        plt.figure(figsize=(10, 6))
        plt.plot(freq_array, actual_te, label='Actual TE (from CST)', color='b')
        plt.plot(freq_array, desired_te, label='Desired TE (Ideal)', color='r', linestyle='--')
        
        plt.title(f'TE Response Comparison for Matrix {matrix_number}')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('TE Magnitude (linear)')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.ylim(-0.1, 1.1) # Set y-axis limits for better visualization
        
        # Add the MSE value as text on the plot
        plt.text(0.05, 0.95, f'MSE: {mse:.6f}', 
                 transform=plt.gca().transAxes, 
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        
        # Save the plot to a file
        plot_filename = f"matrix_{matrix_number}_response.png"
        save_path = os.path.join(output_dir, plot_filename)
        plt.savefig(save_path)
        plt.close() # Close the figure to free up memory
        
        print(f"Plot saved to: {os.path.abspath(save_path)}")

    print("-" * 50)
    print("All matrices processed.")
