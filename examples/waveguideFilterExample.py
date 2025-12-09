# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

#%%
import numpy as np
import matplotlib.pyplot as plt
import context
import cst_python_api as cpa


projectName = "Waveguide_Filter_Example"

# Create the CST project
myCST = cpa.CST_MicrowaveStudio(context.dataFolder, projectName + ".cst")

# Set the default units for the project
myCST.Project.setUnits()

######################
#
# CREATE THE 3D MODEL
#
######################
#%%
# Definition of the geometry parameters for the antenna
f0 = 17.0
lambda0 = 300/f0
lPort = lambda0/2

aWG = 10.0
bWG = 5.0
vIris = [8.6, 7.7, 7.3, 7.2, 7.2, 7.3, 7.7, 8.6]
vRes = [11.7, 13.1, 13.7, 13.8, 13.7, 13.1, 11.7]
tIris = 2.0

currentZPos = 0.0
idxPort = 0
idxIris = 0
idxRes = 0
for ii in range(len(vIris) + len(vRes) + 2):
    
    # If the current element is the first or last one, create a port section
    if ii == 0 or ii == (len(vIris) + len(vRes) + 1):
        # Create the resonator
        myCST.Build.Shape.addBrick(
            xMin = -aWG/2, xMax = aWG/2,
            yMin = -bWG/2, yMax = bWG/2,
            zMin = currentZPos, zMax = currentZPos + lPort,
            name = "Port{:d}".format(idxPort+1), component = "component1",
            material="Vacuum"
        )
        # Update the current z position
        currentZPos = currentZPos + lPort
        # Increment the counter of ports
        idxPort = idxPort + 1
        
    # If ii is odd, then an iris must be created
    elif ii % 2 != 0:
        # Create the iris
        myCST.Build.Shape.addBrick(
            xMin = -vIris[idxIris]/2, xMax = vIris[idxIris]/2,
            yMin = -bWG/2, yMax = bWG/2,
            zMin = currentZPos, zMax = currentZPos + tIris,
            name = "Iris{:d}".format(idxIris+1), component = "component1",
            material="Vacuum"
        )
        # Update the current z position
        currentZPos = currentZPos + tIris
        # Increment the counter of irises
        idxIris = idxIris + 1
        
    # If ii is even, then a resonator must be created
    else:
        # Create the resonator
        myCST.Build.Shape.addBrick(
            xMin = -aWG/2, xMax = aWG/2,
            yMin = -bWG/2, yMax = bWG/2,
            zMin = currentZPos, zMax = currentZPos + vRes[idxRes],
            name = "Resonator{:d}".format(idxRes+1), component = "component1",
            material="Vacuum"
        )
        # Update the current z position
        currentZPos = currentZPos + vRes[idxRes]
        # Increment the counter of resonators
        idxRes = idxRes + 1
        

###########################
#
# CONFIGURE THE SIMULATION
#
###########################

myCST.Solver.setFrequencyRange(16.0, 18.0)

# Set the bounding box limits
myCST.Solver.setBackgroundLimits(
    xMin = 0.0, xMax = 0.0,
    yMin = 0.0, yMax = 0.0,
    zMin = 0.0, zMax = 0.0,
)

# Set the bakcground material
myCST.Solver.setBackgroundMaterial("PEC")

# Set the boundary conditions
myCST.Solver.setBoundaryCondition(
    xMin = "electric", xMax = "electric",
    yMin = "electric", yMax = "electric",
    zMin = "electric", zMax = "electric",
)

# Set the symmetry planes (EHMV)
myCST.Solver.addSymmetryPlane(
    xSym = "magnetic", ySym = "electric", zSym = "none")

# Define excitation ports
myCST.Solver.Port.addWaveguidePort(
    xMin =-aWG/2, xMax = aWG/2,
    yMin = -bWG/2, yMax = bWG/2,
    zMin = 0.0, zMax = 0.0,
    orientation = "zmin", nModes = 1
)
myCST.Solver.Port.addWaveguidePort(
    xMin =-aWG/2, xMax = aWG/2,
    yMin = -bWG/2, yMax = bWG/2,
    zMin = currentZPos, zMax = currentZPos,
    orientation = "zmax", nModes = 1
)

# Set the solver type
myCST.Solver.changeSolverType("HF Frequency Domain")

##############################################
#
# RUN THE SIMULATION AND RETRIEVE THE RESULTS
#
##############################################
#%%
# Launch the simulation
myCST.Solver.runSimulation()

#%%
# Retrieve S-Parameter results
freq, S11 = myCST.Results.getSParameters(1, 1)
_, S21 = myCST.Results.getSParameters(2, 1)

#%%
# PLOT THE S-PARAMETERS RESULTS
# Create a new figure
plt.figure(0)
# Plot the retrieved results
plt.plot(freq, 20*np.log10(np.abs(S11)), color="blue", label="$S_{11}$")
plt.plot(freq, 20*np.log10(np.abs(S21)), color="red", label="$S_{21}$")
# Decorate the figure
plt.xlabel("Frequency (GHz)")
plt.ylabel("$|S_{ij}|$ (dB)")
plt.legend()
plt.grid()
plt.show()