# Optimizing Dynamic Programming-Based Algorithms
This is an on-going project in an attempt to make solving grid-based dynamic programming problems, 
such as Hamilton-Jacobi partial differential equations (HJ PDE) and table-based value iteration, easier and faster.

# Quickstart (Ubuntu 18 & 20)
Please install the following:
* Install Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
* Install heteroCL (http://heterocl.csl.cornell.edu/doc/installation.html):

    Create new conda environment: ``` conda create --name hcl-env ```
    
    Activate new conda environment: ``` conda activate hcl-env ```
    
    Install pre-built heterocl: ``` conda install -c cornell-zhang heterocl -c conda-forge```
    
* Potential Python modules to install if you run into errors:

    ``` pip3 install future plotly==4.5.0 ```
    
* Run the example code from optimized_dp root directory: ``` python3 user_definer.py ```
* Note: If you're on Ubuntu 20.04, you may have encounter an error regarding ``` libtinfo5 ```. To fix,
please just run this command ```sudo apt-install libtinfo5 ``` 

# Update 31/Jan/2021
The repo has been updated to allow multiple instances of problem to be defined. HJSolver are called and
parameters are passed into this function for problem to be solved. 

# Update 16/Sept/2020
The repo currently works for 3, 4, 5, 6 dimension systems on CPU. 6 dimension graph currently does 
support disturbance with maximum of 4 control inputs.
Notes: For 6 dimensions, recommended grid size is 20-30 each dimension on system with 32Gbs of DRAM.

# Dependencies
These instructions have been tested and work on Ubuntu 18.04. Please install the following:
* Anaconda (https://docs.anaconda.com/anaconda/install/linux/)
* HeteroCL (http://heterocl.csl.cornell.edu/doc/installation.html) ``` conda create --name hcl-env ```
    ``` conda activate hcl-env ```. Install pre-built heterocl ``` conda install -c cornell-zhang heterocl -c conda-forge```. 
  Note: You might also need to install module "future" if encounter an error, with this command ``` pip3 install future ```.  
*  Plotly. This library for visualization. ``` $pip3 install plotly==4.5.0 ```

# Current code structure explanation
* user_definer.py: specify grid numbers, dynamic systems intialization, initial value function ,computation method.
* solver.py: Compute HJ PDE, the end result is V1 value function
* dynamics/ : User's dynamical system specification
* Shapes/ShapesFunctions.py : Add-in functions for calculating different intial value functions
* computeGraphs/CustomGraphFunctions.py: Ready-to-user HeteroCL style utility functions

# Tips to specify your own problem and use the final result
* Create a class file in folder dynamics/ to specify your system characteristics
* Use user_definer.py to specify grid, the object and computation method (ex. "minVWithV0": calculating tube). 
  Remember to import the file
created earlier in user_definer.py
* For large dimensional system (greater than 3), you may want to save the final value function for processing. 
  In the file solver.py, at line 116, fill in anycodes to save it.
* If you want to save value functions to disk by time step, fill in any code to save it at around line 108 inside the while loop.

# Related Projects
## MATLAB
* [A Toolbox of Level Set Methods ](https://www.cs.ubc.ca/~mitchell/ToolboxLS/)
* [helperOC](https://github.com/HJReachability/helperOC)
## C++
* [BEACLS](https://hjreachability.github.io/beacls/)
## Python
* [hj_reachability](https://github.com/StanfordASL/hj_reachability)