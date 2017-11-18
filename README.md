# drag_module
This is a proposed module for implementing a variety of Drag method style transition state searches through the atomic simulation environment (ASE).
These are all methods based on somehow constructing a trial reaction path between the supplied initial and final states of the reaction and then relaxing these images to obtain an estimate of the true transition state.

AidanDrag.py is the code for the module itself. Details of the inputs, outputs, and variables are included in the code itself.
In brief, the module requires initial and final state trajectory files, a number of intermediate images to create, a reaction coordinate to relax along, and a calculator object to actually perform the relaxations.
Users can also choose if the path is created via even sampling or a bisection sampling, and if the path is created using linear interpolation or using IDPP interpolation (Smidstrup, et al. JCP 140, 214106 (2014)).
The outputs include trajectory image files and a numpy (.npz) data file.

A minimum working example of NH3 dissociation on a Ag fcc111 surface is included as Ag_NH2.MWE.py, along with necessary initial and final state trajectory files.
The cluster submission line at the top of the file will need to be changed to fit the user's computing environment.

A slighly more complicated template file for using this module is included as module_drag_template.py. The results of using this file for the NH3 dissociation on Ag(111) are included in the folder full_example.

Aidan J. Klobuchar
