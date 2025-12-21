

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
        
#         myCST.Build.Material.addNormalMaterial("FR4 (Lossy)", 4.3, 1.0, colour=[0.94, 0.82, 0.76])
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
#         save_path = r"C:/Users/IDARE_ECE/Documents/saved_cst_projects/"
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