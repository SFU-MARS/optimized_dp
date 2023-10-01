from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class PlotOptions:
    def __init__(self, do_plot=True, plot_type="3d_plot", plotDims=[], slicesCut=[], min_isosurface = 0, max_isosurface = 0):
        if plot_type not in ["2d_plot", "3d_plot"]:
            raise Exception("Illegal plot type !")

        if plot_type == "2d_plot" :
            if len(plotDims) != 2:
                raise Exception("Make sure that dim_plot size is 2 !!")

        if plot_type == "3d_plot" :
            if len(plotDims) != 3:
                raise Exception("Make sure that dim_plot size is 3 !!")

        self.do_plot = do_plot
        self.dims_plot = plotDims
        self.plot_type = plot_type
        self.slices = slicesCut
        self.min_isosurface = min_isosurface
        self.max_isosurface = max_isosurface
        

def plot_isosurface(grid, V, plot_option):
    dims_plot = plot_option.dims_plot
    idx = [slice(None)] * grid.ndims
    slice_idx = 0

    dims_list = list(range(grid.ndims))
    for i in dims_list:
        if i not in dims_plot:
            idx[i] = plot_option.slices[slice_idx]
            slice_idx += 1

    if len(dims_plot) != 3 and len(dims_plot) != 2:
        raise Exception('dims_plot length should be equal to 3\n')

    if len(dims_plot) == 3:
        # Plot 3D isosurface
        dim1, dim2, dim3 = dims_plot[0], dims_plot[1], dims_plot[2]
        complex_x = complex(0, grid.shape[dim1])
        complex_y = complex(0, grid.shape[dim2])
        complex_z = complex(0, grid.shape[dim3])
        mg_X, mg_Y, mg_Z = np.mgrid[grid.min_bounds[dim1]:grid.max_bounds[dim1]:complex_x,
                                    grid.min_bounds[dim2]:grid.max_bounds[dim2]:complex_y,
                                    grid.min_bounds[dim3]:grid.max_bounds[dim3]:complex_z]

        my_V = V[tuple(idx)]

        if (my_V > 0.0).all():
            print("Implicit surface will not be shown since all values are positive ")
        if (my_V < 0.0).all():
            print("Implicit surface will not be shown since all values are negative ")

        print("Plotting beautiful plots. Please wait\n")
        fig = go.Figure(data=go.Isosurface(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=mg_Z.flatten(),
            value=my_V.flatten(),
            colorscale='jet',
            isomin=plot_option.min_isosurface,
            surface_count=1,
            isomax=plot_option.max_isosurface,
            caps=dict(x_show=True, y_show=True)
        ))
        fig.show()
        print("Please check the plot on your browser.")

    if len(dims_plot) == 2:
        dim1, dim2 = dims_plot[0], dims_plot[1]
        complex_x = complex(0, grid.shape[dim1])
        complex_y = complex(0, grid.shape[dim2])
        mg_X, mg_Y = np.mgrid[grid.min_bounds[dim1]:grid.max_bounds[dim1]:complex_x,
                              grid.min_bounds[dim2]:grid.max_bounds[dim2]:complex_y]

        my_V = V[tuple(idx)]

        if (my_V > 0.0).all():
            print("Implicit surface will not be shown since all values are positive ")
        if (my_V < 0.0).all():
            print("Implicit surface will not be shown since all values are negative ")

        print("Plotting beautiful 2D plots. Please wait\n")
        fig = go.Figure(data=go.Contour(
            x=mg_X.flatten(),
            y=mg_Y.flatten(),
            z=my_V.flatten(),
            zmin=0.0,
            ncontours=1,
            contours_coloring='none',  # former: lines
            name="Reachable Set",  # zero level
            line_width=1.5,
            line_color='magenta',
            zmax=0.0,
        ), layout=go.Layout(plot_bgcolor='rgba(0,0,0,0)'))  # ,paper_bgcolor='rgba(0,0,0,0)'

        fig.show()
        print("Please check the plot on your browser.")

def hj_plot(val_func, grid, steer_idx, vel_idx, time_idx):
    po = PlotOptions(do_plot=True,
                     plot_type="3d_plot",
                     plotDims=[0,1, 2],
                     slicesCut=[steer_idx, vel_idx])
    plot_isosurface(grid, val_func[:, :, :, :, :, time_idx], plot_option = po)


def xy_plot(val_func, grid, heading_idx, steer_idx, vel_idx, time_idx, cmap="Purples"):
    values = deepcopy(val_func[:, :, heading_idx, steer_idx, vel_idx, time_idx])
    values[values>0.0] = float("nan")
    values = -values
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.gca().invert_yaxis()
    plt.imshow(values, cmap=cmap, alpha=1)