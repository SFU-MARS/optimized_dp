# Optimizing Dynamic Programming-Based Algorithms
This is an on-going project in an attempt to make solving grid-based dynamic programming problems, such as Hamilton-Jacobi partial differential equations (HJ PDE) and table-based value iteration, easier and faster on all available hardware platforms.

# Update 04/June/2020
The repo currently works for 4, 5, 6 dimension systems on CPU. 6 dimension graph currently does not support disturbance with disturbances. 

# Dependencies
* HeteroCL (http://heterocl.csl.cornell.edu/doc/installation.html) - A Python-based domain-specific language (DSL). The HeteroCL DSL provides a clean abstraction that decouples algorithm specification from three important types of hardware customization in compute, data types, and memory architectures. Follow the link to install HeteroCL.
*  Plotly. This library for visualization. ``` $pip install plotly==4.5.0 ```

# Current code structure explanation
* user_definer.py: specify grid numbers, dynamic systems intialization, initial value function ,computation method.
* solver.py: Compute HJ PDE, the end result is V1 value function
* dynamics/ : User's dynamical system specification
* Shapes/ShapesFunctions.py : Add-in functions for calculating different intial value functions
* computeGraphs/CustomGraphFunctions.py: Ready-to-user HeteroCL style utility functions

# Tips to specify your own problem and use the final result
* Create a class file in folder dynamics/ to specify your system characteristics
* Use user_definer.py to specify grid, the object and computation method (ex. "minVWithVInit": calculating tube). Remember to import the file
created earlier in user_definer.py
* For large dimensional system (greater than 3), you may want to save the final value function for processing. In the file solver.py, at line 116, fill in anycodes to save it.
* If you want to save value functions to disk by time step, fill in any code to save it at around line 108 inside the while loop.
# Running
``` python3 solver.py ```
