# Optimizing Dynamic Programming-Based Algorithms
This is an on going project in an attempt to solve Hamilton Jacobian partial differential equation (HJ PDE) easier and faster on all available hardware platforms.

# Dependencies
* HeteroCL (http://heterocl.csl.cornell.edu/doc/installation.html) - A Python-based domain-specific language (DSL). The HeteroCL DSL provides a clean abstraction that decouples algorithm specification from three important types of hardware customization in compute, data types, and memory architectures. Follow the link to install HeteroCL.
*  Plotly. This library for visualization. ``` $pip install plotly==4.5.0 ```

# Current code structure explanation
* user_definer.py: specify grid numbers, dynamic systems intialization, initial value function ,computation method.
* solver.py: Compute HJ PDE, the end result is V1 value function
* dynamics/ : User's dynamical system specification
* Shapes/ShapesFunctions.py : Add-in functions for calculating different intial value functions
* computeGraphs/CustomGraphFunctions.py: Ready-to-user HeteroCL style utility functions 
# Running
``` python3 solver.py ```
