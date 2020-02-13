# HJ_PDE_heteroCL
This is an on going project in an attempt to solve Hamilton Jacobian partial differential equation (HJ PDE) easier and faster on all available hardware platforms.

# Dependencies
* [HeteroCL] (http://heterocl.csl.cornell.edu/web/) - A Python-based domain-specific language (DSL). The HeteroCL DSL provides a clean abstraction that decouples algorithm specification from three important types of hardware customization in compute, data types, and memory architectures. Follow the link to install HeteroCL.
*  Plotly. This library for visualization. ``` $pip install plotly==4.5.0 ```

# Code base explanation
GridProcessing.py includes api to create a grid structure. ShapesFunctions.py has functions to initilize value functions. HJ_PDE_HeteroCL.py is where HJ pde is solved and visualized.

# Running sample code
``` python3 HJ_PDE_HeteroCL.py ```
