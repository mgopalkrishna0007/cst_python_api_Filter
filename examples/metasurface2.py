import numpy as np
import matplotlib.pyplot as plt
import context
import cst_python_api as cpa
import os
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

# Project settings
projectName = "pixelated metasurface"

# Create the CST project
myCST = cpa.CST_MicrowaveStudio(context.dataFolder, projectName + ".cst")

# Set the default units for the project
myCST.Project.setUnits()
print ("units are set to default")

# Parameters based on MATLAB code
st = 0.8  # thickness of substrate
ls = 5.0   # substrate length
p = 0.0    # pixel thickness (assuming this is height for pixels)

# Add FR4 material to the project
myCST.Build.Material.addNormalMaterial(
    "FR4", 4.3, 1.0, colour=[0.94, 0.82, 0.76] , tanD=0.025)
print ("FR4 material is added to the project")

print ("adding groundplane , substrate and pixels to the project...")
# Create ground plane
# myCST.Build.Shape.addBrick(
#     xMin=-ls/2.0, xMax=ls/2.0,
#     yMin=-ls/2.0, yMax=ls/2.0,
#     zMin=0.0, zMax=-p,
#     name="Groundplane", component="component1", material="PEC"
# )

# Create substrate
myCST.Build.Shape.addBrick(
    xMin=-ls/2.0, xMax=ls/2.0,
    yMin=-ls/2.0, yMax=ls/2.0,
    zMin=0.0, zMax=st,
    name="Substrate", component="component1", material="FR4"
)

# Create individual pixels (patch elements) with specific dimensions
# Pixel 1
myCST.Build.Shape.addBrick(
    xMin=-ls/10.0, xMax=ls/10.0,
    yMin=3.0*ls/10.0, yMax=ls/2.0,
    zMin=st, zMax=st + p,
    name="Pixel1", component="component1", material="PEC"
)

# # Pixel 2
# myCST.Build.Shape.addBrick(
#     xMin=ls/10.0, xMax=3.0*ls/10.0,
#     yMin=3.0*ls/10.0, yMax=ls/2.0,
#     zMin=st, zMax=st + p,
#     name="Pixel2", component="component1", material="PEC"
# )

# Pixel 3
myCST.Build.Shape.addBrick(
    xMin=3.0*ls/10.0, xMax=ls/2.0,
    yMin=3.0*ls/10.0, yMax=ls/2.0,
    zMin=st, zMax=st + p,
    name="Pixel3", component="component1", material="PEC"
)

# Pixel 4
myCST.Build.Shape.addBrick(
    xMin=-ls/10.0, xMax=ls/10.0,
    yMin=ls/10.0, yMax=3.0*ls/10.0,
    zMin=st, zMax=st + p,
    name="Pixel4", component="component1", material="PEC"
)

# Pixel 5
myCST.Build.Shape.addBrick(
    xMin=-ls/10.0, xMax=-3.0*ls/10.0,
    yMin=3.0*ls/10.0, yMax=ls/2.0,
    zMin=st, zMax=st + p,
    name="Pixel5", component="component1", material="PEC"
)

# Pixel 5
myCST.Build.Shape.addBrick(
    xMin=-3*ls/10.0, xMax=-ls/2.0,
    yMin=-ls/10.0, yMax=ls/10.0,
    zMin=st, zMax=st + p,
    name="Pixel6", component="component1", material="PEC"
)

myCST.Build.Shape.addBrick(
    xMin=-ls/10.0, xMax=-3*ls/10.0,
    yMin=-ls/10.0, yMax=ls/10.0,
    zMin=st, zMax=st + p,
    name="Pixel7", component="component1", material="PEC"
)

myCST.Build.Shape.addBrick(
    xMin=ls/10.0, xMax=3*ls/10.0,
    yMin=-ls/10.0, yMax=ls/10.0,
    zMin=st, zMax=st + p,
    name="Pixel8", component="component1", material="PEC"
)

myCST.Build.Shape.addBrick(
    xMin=3*ls/10.0, xMax=ls/2.0,
    yMin=-ls/10.0, yMax=-3*ls/10.0,
    zMin=st, zMax=st + p,
    name="Pixel9", component="component1", material="PEC"
)

myCST.Build.Shape.addBrick(
    xMin=-3*ls/10.0, xMax=-ls/2.0,
    yMin=-ls/10.0, yMax=-3*ls/10.0,
    zMin=st, zMax=st + p,
    name="Pixel10", component="component1", material="PEC"
)

myCST.Build.Shape.addBrick(
    xMin=-ls/10.0, xMax=-3*ls/10.0,
    yMin=-ls/10.0, yMax=-3*ls/10.0,
    zMin=st, zMax=st + p,
    name="Pixel11", component="component1", material="PEC"
)

# Pixel 6
myCST.Build.Shape.addBrick(
    xMin=ls/10.0, xMax=3.0*ls/10.0,
    yMin=-ls/10.0, yMax=-3.0*ls/10.0,
    zMin=st, zMax=st + p,
    name="Pixel12", component="component1", material="PEC"
)

# Pixel 7
myCST.Build.Shape.addBrick(
    xMin=3.0*ls/10.0, xMax=ls/2.0,
    yMin=-ls/10.0, yMax=ls/10.0,
    zMin=st, zMax=st + p,
    name="Pixel13", component="component1", material="PEC"
)

# Pixel 8
myCST.Build.Shape.addBrick(
    xMin=-ls/10.0, xMax=-3.0*ls/10.0,
    yMin=-3.0*ls/10.0, yMax=-ls/2.0,
    zMin=st, zMax=st + p,
    name="Pixel14", component="component1", material="PEC"
)

# Pixel 9
myCST.Build.Shape.addBrick(
    xMin=-3.0*ls/10.0, xMax=-ls/2.0,
    yMin=ls/10.0, yMax=3.0*ls/10.0,
    zMin=st, zMax=st + p,
    name="Pixel15", component="component1", material="PEC"
)
print ("groundplane , substrate and pixels are added to the project succesfully")

# Set frequency range for simulation
myCST.Solver.setFrequencyRange(14.0, 16.0)
print ("frequency range is set for simulation")

# Set the bounding box limits
myCST.Solver.setBackgroundLimits(
    xMin = 0.0, xMax = 0.0,
    yMin = 0.0, yMax = 0.0,
    zMin = 0.0, zMax = 0.0,
)

print ("bounding box limits are set")

# Set the boundary conditions
myCST.Solver.setBoundaryCondition(
    xMin = "unit cell", xMax = "unit cell",
    yMin = "unit cell", yMax = "unit cell",
    zMin = "expanded open", zMax = "expanded open",
)

print("boundary conditions are set")

myCST.Solver.defineFloquetModes(nModes=2, theta=0.0, phi=0.0, forcePolar=False, polarAngle=0.0)

print("floquet modes are defined")

# Change solver type
myCST.Solver.changeSolverType("HF Frequency Domain")
myCST.Solver.SetNumberOfResultDataSamples(501)

print(dir(myCST.Solver))  # Is `SetNumberOfResultDataSamples` listed?
print(type(myCST.Solver))  # Should show a COM wrapper class

print("solver type is changed to HF Frequency Domain")
print("running simulation...")
# Run the simulation
myCST.Solver.runSimulation()
print("simulation is completed")

print("retrieving S-parameters...")
# Retrieve the S-Parameters results
freq, SZMax1ZMax1 = myCST.Results.getSParameters(0, 0, 1, 1)
_, SZMax2ZMax1 = myCST.Results.getSParameters(0, 0, 2, 1)
_,SZMin1ZMax1 = myCST.Results.getSParameters(-1, 0, 1, 1)
_,SZMin2ZMax1 = myCST.Results.getSParameters(-1, 0, 2, 1)
_,SZMax1ZMax2 = myCST.Results.getSParameters(0, 0, 1, 2)
_,SZMax2ZMax2 = myCST.Results.getSParameters(0, 0, 2, 2)
_,SZMin1ZMax2 = myCST.Results.getSParameters(-1, 0, 1, 2)
_,SZMin2ZMax2 = myCST.Results.getSParameters(-1, 0, 2, 2)
_,SZMax1ZMin1 = myCST.Results.getSParameters(0, -1, 1, 1)
_,SZMax2ZMin1 = myCST.Results.getSParameters(0, -1, 2, 1)       
_,SZMin1ZMin1 = myCST.Results.getSParameters(-1, -1, 1, 1)
_,SZMin2ZMin1 = myCST.Results.getSParameters(-1, -1, 2, 1)
_,SZMax1ZMin2 = myCST.Results.getSParameters(0, -1, 1, 2)
_,SZMax2ZMin2 = myCST.Results.getSParameters(0, -1, 2, 2)
_,SZMin1ZMin2 = myCST.Results.getSParameters(-1, -1, 1, 2)
_,SZMin2ZMin2 = myCST.Results.getSParameters(-1, -1, 2, 2)

print("S-parameters are retrieved")
print("freq = " ,freq)
print (len(freq))
print("SZMin2ZMax2 = ", SZMin2ZMax2)
print("SZMax2ZMax2 = ", SZMax2ZMax2)
# print("SZMin2ZMax2 = ", SZMin2ZMax2)
# print("SZMax2ZMax2 = ", SZMax2ZMax2)
print (len(SZMin2ZMax2))
print (len(SZMax2ZMax2))
# print (len(SZMin2ZMax2))
# print (len(SZMax2ZMax2))

print("plotting S-parameters...")

# Export S-parameters to files
exportpathA = r'C:\Users\GOPAL\OneDrive\Documents\MATLAB\pixel2.txt'
exportpathB = exportpathA.replace('.txt', '_set2.txt')
exportpathC = exportpathA.replace('.txt', '_set3.txt')
exportpathD = exportpathA.replace('.txt', '_set4.txt')
exportpathE = exportpathA.replace('.txt', '_set5.txt')
exportpathF = exportpathA.replace('.txt', '_set6.txt')
exportpathG = exportpathA.replace('.txt', '_set7.txt')
exportpathH = exportpathA.replace('.txt', '_set8.txt')
exportpathI = exportpathA.replace('.txt', '_set9.txt')
exportpathJ = exportpathA.replace('.txt', '_set10.txt')
print("exporting S-parameters to files...")
def export_s_parameters(filepath, frequency, s_parameters):
    with open(filepath, 'w') as f:
        for freq_val, s_param in zip(frequency, s_parameters):
            # f.write(f"{freq_val}\t{20 * np.log10(np.abs(s_param))}\n")  # Convert to dB
            f.write(f"{freq_val}\t{(np.abs(s_param))}\n")  # Convert to dB
# export_s_parameters(exportpathA, freq, SZMax2ZMin2)
# export_s_parameters(exportpathB, freq, SZMin2ZMin2)   
# export_s_parameters(exportpathC, freq, SZMin1ZMin1)
# export_s_parameters(exportpathD, freq, SZMin2ZMin1)
# export_s_parameters(exportpathE, freq, SZMax1ZMin1)

export_s_parameters(exportpathA, freq, SZMin2ZMax2)
export_s_parameters(exportpathB, freq, SZMax2ZMax2)   
export_s_parameters(exportpathC, freq, SZMax1ZMax1)
export_s_parameters(exportpathD, freq, SZMax2ZMax1)
export_s_parameters(exportpathE, freq, SZMin1ZMax1)

# export_s_parameters(exportpathF, freq, SZMin2ZMax1)
# export_s_parameters(exportpathG, freq, SZMax1ZMax2)
# export_s_parameters(exportpathH, freq, SZMax2ZMax2)
# export_s_parameters(exportpathI, freq, SZMin1ZMax2)
# export_s_parameters(exportpathJ, freq, SZMin2ZMax2)


# Load and plot results from both files if they exist
if os.path.isfile(exportpathA):
    FrequencyA, SparameterA = np.loadtxt(exportpathA, unpack=True)

    if os.path.isfile(exportpathB):
        FrequencyB, SparameterB = np.loadtxt(exportpathB, unpack=True)
        
        if os.path.isfile(exportpathD):
            FrequencyC, SparameterC = np.loadtxt(exportpathC, unpack=True)
            
            if os.path.isfile(exportpathC):
                FrequencyE, SparameterE = np.loadtxt(exportpathE, unpack=True)
                
                # # Plot results from both files
                # plt.figure()
                # plt.plot(FrequencyA, SparameterA, '-', linewidth=2, label='SZMin2ZMax2', color='blue')  
                # plt.plot(FrequencyB, SparameterB, '-', linewidth=2, label='SZMax2ZMax2', color='red')
                # plt.plot(FrequencyC, SparameterC, '-', linewidth=2, label='SZMax1ZMax1', color='green')
                # plt.plot(FrequencyE, SparameterE, '-', linewidth=2, label='SZMin1ZMax1', color='orange')
                # plt.grid()
                # plt.legend(loc='best')
                # plt.title('S Parameters')
                # plt.xlabel('Frequency (GHz)')
                # plt.ylabel('(dB)')

        MAG_A = ((SparameterE))**2# Convert dB to magnitude for A
        MAG_B = ((SparameterC))**2  # Convert dB to magnitude for B
                   
        def resultCalculation(Frequency, MAG_A, MAG_B):
            # Calculate expression and invert it 
            result = (MAG_A) / ((MAG_A) + (MAG_B))
            print ("max and min : " , max(result), min(result)
            )            
            # # Plot the inverted result
            # plt.figure()
            # plt.plot(Frequency, result, linewidth=2)
            # plt.grid()
            # plt.title('Mixed Template results')
            # plt.xlabel('Frequency (GHz)')
            # plt.ylabel(' ')
            # # plt.ylim([0.5, 1])  # Set y-axis limits to be linear from 0 to 1
            # plt.show()
            
            std_dev = 1
            x1 = np.arange(14.0,16.1, 0.1)
            len1 = len(x1)
            len2 = len(FrequencyA)
            interp_indices = np.linspace(0, len1 - 1, len2 - 2)
            interp_func = interp1d(np.arange(len1), x1)
            interpolated = interp_func(interp_indices)
            extended_x1 = np.concatenate(([x1[0]], interpolated, [x1[-1]]))
            y1 = np.exp(-((extended_x1 - 15.0)**2) / (2 * std_dev**2))
            print ( "len of extended_x1 : " , len(extended_x1))
            print ( "len of y1 : ", len(y1))
            print ( "len of FrequencyA : ", len(FrequencyA))
            print ("len of result :" , len(result))
            cost = mean_squared_error(y1, result)
            plt.figure(figsize=(8,4))
            plt.plot(FrequencyA, result, 'b-', label='Actual TE (Simulated)')
            plt.plot(extended_x1, y1, 'r--', label='Target TE (Gaussian)')
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('Transmission Efficiency')
            plt.title(f'TE Comparison (Cost: {cost:.4f})')
            plt.legend()
            plt.grid(True)
            plt.xlim(freq[0], freq[-1])
            plt.ylim(0.6, 1.)
            plt.show()
            plt.pause(0.1)


        resultCalculation(FrequencyA, MAG_A, MAG_B)
    else:
        raise FileNotFoundError(f'The file {exportpathB} does not exist.')
else:
    raise FileNotFoundError(f'The file {exportpathA} does not exist.')

