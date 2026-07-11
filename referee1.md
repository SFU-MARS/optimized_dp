Reviewer Comments to Author:
Referee: 1

Recommendation: Reject

Comments:
Summary
=======

This paper describes the OptimizedDP python package for solving Hamilton-Jacobi partial differential equations (HJ-PDEs) and continuous-space Markov decision processes (MDPs) with grid-based iterative algorithms. The package is built on the HeteroCL programming stack which allows very fast execution, and, surprisingly, it is shown to solve problems with up to 8 dimensions on a high-performance machine with 1 TB RAM (these types of problems are usually limited to about 5 dimensions, and 8 is typically considered intractable). Although the package of course does not break the curse of dimensionality, the optimizations it does contain, including innovative ways to avoid storing large grids of temporary variables, allow solving larger problems than previously possible.

Evaluation
==========

OptimizedDP meaningfully advances the state of the art. I have not seen an 8 dimensional HJ-PDE problem solved before, and that is an accomplishment. However, there are some significant deficiencies in the package. The paper is generally well-written, but there are also some significant issues noted below. The issues outlined below should be addressed before publication.

Major Comments that should be addressed before publication
==========================================================

1. The package does not appear to have any tests (there is a `test_mdp_3d.py` file, but it does not appear to work). Tests are crucial for maintenance of any open source package, and, in my opinion, should be a requirement for publication in this journal. The authors may have intended the examples to function as tests, but there are many reasons that examples are poor tests, including the time they take to run, lack of automation, etc. Moreover, I could not get all of the examples to run. For instance `plotting_examples.py` failed after several minutes with `NameError: name 'secondOrderX' is not defined`
2. The online documentation for the package appears to be limited to the README and examples, which are far from comprehensive. A numerical package like this may not need comprehensive documentation, but if one searches for documentation, they might find `odp/MDP_example/Example_3D.py`, which appears to be outdated and incorrect! Outdated documentation must be removed and some correct documentation provided.
3. My main confusion in the paper is the statement (which is repeated several several times) "As we increase the number of dimensions, the DRAM memory required for these temporary array goes up linearly." Initially, I thought this was a typo, since, if we use the same number of divisions per dimension, the number of grid elements increases *exponentially* with the dimension. Figure 2 seems to show that "N arrays" are created for the random variables, which does imply linearly more memory, however, each of these arrays are the size of the grid, which increases exponentially, so I am not sure why the linear increase is important. This should be explained more clearly explained.

Minor Comments
==============

Installation and Running
------------------------
- Installation requires manual installation of an old library, `libtinfo5`. This seems to be deprecated in favor of `libtinfo6`
- The `odp/dynamics/__init__.py` file appears to be incorrect because, at minimum, `DubinsCar5D.py` does not exist.

README
------
I had the following questions when trying to interpret the starter code on the readme:
- Why is it necessary to provide a `dims` argument. Can't this be inferred from the grid arguments in the constructor.
- Do `grid_min` and `grid_max` need to be numpy arrays or will lists work?
- What is `pd`?
- `examples/plotting_example.py` link is dead

Additional Comments
===================

These comments do not necessarily need to be addressed before publication, but are comments on the approach:
- One shortcoming of the interface is that the user defining the dynamics needs to write Hetero CL code, e.g. `hcl.scalar`, so they need to understand at least part of the Hetero CL package, and cannot just write python. In my opinion, this is a shortcoming of python, and we should consider using languages designed for numerical computing in the future.
- The HJ-PDE interface requires the *user* to implement a `opt_ctrl` function that returns the optimal control. It seems like the user should only need to specify dynamics, and the package should automatically calculate the optimal control. I realize that this pattern is common in HJ-PDE libraries (e.g. I believe it is also in the HelperOC package for instance), but it is difficult to understand why this can't be done by the package, at least for simple cases.
- The `transition` function in the MDP interface returns a matrix where the first column is the probability and subsequent columns define the possible new states. Besides being confusing by mixing probabilities with state values, this limits the support of the transition distribution to be a set with finite cardinality, precluding many distributions often used in MDPs such as Gaussians.
- In the MDP interface, the `transition` function takes an `iVals` argument. In the documentation I found, I was not able to determine exactly what `iVals` is, but I assume that it is some type of index that has to do with the grid. Interfaces involving both the state space and the index space seem very prone to errors and confusion. I think it would be much better to have the user write code entirely in terms of the state values OR in terms of index values.

Additional Questions:
Please help ACM create a more efficient time-to-publication process: Using your best judgment, what amount of copy editing do you think this paper needs?: Moderate

Most ACM journal papers are researcher-oriented. Is this paper of potential interest to developers and engineers?: Yes