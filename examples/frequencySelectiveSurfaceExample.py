# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

#%%
import numpy as np
import matplotlib.pyplot as plt
import context
import cst_python_api as cpa


projectName = "Frequency_Selective_Surface_Example"

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
# Definition of the geometry parameters for the FSS
f0 = 10.0 # Central frquency (GHz)
lambda0 = 300/f0 # Wavelength (mm)
Px = 0.6*lambda0 # Periodicity along x (mm)
Py = Px # Periodicity along y (mm)

wSlot = 1.0 # Width of the perforations (mm)
lSlot1 = 14.875 # Length of the perforation of the first screen (mm)
lSlot2 = 14.5 # Length of the perforation of the second screen (mm)
dist12 = 10.0 # Distance between screens 1 and 2 (mm)

screenTh = 0.5 # Thickness of the screens (mm)

# Create screen 1
myCST.Build.Shape.addBrick(
    xMin = -0.5*Px, xMax = 0.5*Px,
    yMin = -0.5*Py, yMax = 0.5*Py,
    zMin = -(dist12/2+screenTh), zMax = -dist12/2,
    name = "Metal", component = "component1/Screen1", material="PEC"
)

# Create horizontal perforation of screen 1
myCST.Build.Shape.addBrick(
    xMin = -0.5*lSlot1, xMax = 0.5*lSlot1,
    yMin = -0.5*wSlot, yMax = 0.5*wSlot,
    zMin = -(dist12/2+screenTh), zMax = -dist12/2,
    name = "Perforation", component = "component1/Screen1", material="Vacuum"
)

# Create vertical perforation of screen 1
myCST.Build.Transform.rotate(
    object="component1/Screen1:Perforation",
    x = 0.0, y = 0.0, z = 90.0,
    copy=True, group = True)

# Insert perforation into screen 1
myCST.Build.Boolean.insert(
    "component1/Screen1:Metal", "component1/Screen1:Perforation")

# Create screen 2
myCST.Build.Shape.addBrick(
    xMin = -0.5*Px, xMax = 0.5*Px,
    yMin = -0.5*Py, yMax = 0.5*Py,
    zMin = dist12/2, zMax = (dist12/2+screenTh),
    name = "Metal", component = "component1/Screen2", material="PEC"
)

# Create horizontal perforation of screen 2
myCST.Build.Shape.addBrick(
    xMin = -0.5*lSlot2, xMax = 0.5*lSlot2,
    yMin = -0.5*wSlot, yMax = 0.5*wSlot,
    zMin = dist12/2, zMax = (dist12/2+screenTh),
    name = "Perforation", component = "component1/Screen2", material="Vacuum"
)

# Create vertical perforation of screen 2
myCST.Build.Transform.rotate(
    object = "component1/Screen2:Perforation",
    x = 0.0, y = 0.0, z = 90.0,
    copy = True, group = True)

# Insert perforation into screen 2
myCST.Build.Boolean.insert(
    "component1/Screen2:Metal", "component1/Screen2:Perforation")

#%%
###########################
#
# CONFIGURE THE SIMULATION
#
###########################

myCST.Solver.setFrequencyRange(5.0, 15.0)

# Set the bounding box limits
myCST.Solver.setBackgroundLimits(
    xMin = 0.0, xMax = 0.0,
    yMin = 0.0, yMax = 0.0,
    zMin = 0.0, zMax = 0.0,
)

# Set the boundary conditions
myCST.Solver.setBoundaryCondition(
    xMin = "unit cell", xMax = "unit cell",
    yMin = "unit cell", yMax = "unit cell",
    zMin = "expanded open", zMax = "expanded open",
)

# Currently it is not possible to adjust the Floquet port options, so the
# default values assigned by CST are used. It should be taken into account that
# 18 modes are used, which will produce a long simulation.

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
# Retrieve the S-Parameters results
freq, SMin1Min1 = myCST.Results.getSParameters(-1,-1,1,1)
_,    SMin2Min1 = myCST.Results.getSParameters(-1,-1,2,1)
_,    SMax1Min1 = myCST.Results.getSParameters(0,-1,1,1)
_,    SMax2Min1 = myCST.Results.getSParameters(0,-1,2,1)

#%%
# PLOT THE S-PARAMETERS RESULTS
# Create a new figure
plt.figure(0)
# Plot the retrieved results
plt.plot(freq, 20*np.log10(np.abs(SMin1Min1)), color="C0", label="Zmin(1),Zmin(1)")
plt.plot(freq, 20*np.log10(np.abs(SMin2Min1)), color="C1", label="Zmin(2),Zmin(1)")
plt.plot(freq, 20*np.log10(np.abs(SMax1Min1)), color="C2", label="Zmax(1),Zmin(1)")
plt.plot(freq, 20*np.log10(np.abs(SMax2Min1)), color="C3", label="Zmax(2),Zmin(1)")
# Decorate the figure
plt.xlabel("Frequency (GHz)")
plt.ylabel("$|S_{ij}|$ (dB)")
plt.legend()
plt.grid()
plt.show()