from dynamics.compute_opt_traj import compute_opt_traj
import numpy as np
from Grid.GridProcessing import Grid
from Shapes.ShapesFunctions import CylinderShape
from plot_options import PlotOptions
from Plots.plotting_utilities import plot_trajectory
from dynamics.DubinsCar import DubinsCar
from solver import HJSolver


def main():
    grid = Grid(np.array([-5.0, -5.0, -np.pi]), np.array([5.0, 5.0, np.pi]), 3, np.array([30, 30, 30]), [2])
    dyn_sys = DubinsCar()
    Initial_value_f = CylinderShape(grid, [2], np.zeros(3), 1)

    lookback_length = 2.0
    t_step = 0.05
    small_number = 1e-5
    tau = np.arange(start=0, stop=lookback_length + small_number, step=t_step)

    comp_method = {"PrevSetsMode": "minVWithV0"}
    V_all_t = HJSolver(dyn_sys, grid, Initial_value_f, tau, comp_method, PlotOptions("3d_plot", [0, 1, 2], []),
                       save_all_t=True)

    test_dyn_sys = DubinsCar(x=np.array([1.5, 1.5, -np.pi]))

    traj, _, _ = compute_opt_traj(grid, V_all_t, tau, test_dyn_sys)
    plot_trajectory(grid, V_all_t, traj)


if __name__ in "__main__":
    main()
