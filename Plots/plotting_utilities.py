import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def plot_isosurface(grid, V, plot_option):
    dims_plot = plot_option.dims_plot
    idx = [slice(None)] * grid.dims
    slice_idx = 0

    dims_list = list(range(grid.dims))
    for i in dims_list:
        if i not in dims_plot:
            idx[i] = plot_option.slices[slice_idx]
            slice_idx += 1


    if len(dims_plot) != 3:
        raise Exception('dims_plot length should be equal to 3\n')
    else:
        dim1, dim2, dim3 = dims_plot[0], dims_plot[1], dims_plot[2]
        complex_x = complex(0, grid.pts_each_dim[dim1])
        complex_y = complex(0, grid.pts_each_dim[dim2])
        complex_z = complex(0, grid.pts_each_dim[dim3])
        mg_X, mg_Y, mg_Z = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]: complex_y,
                                    grid.min[dim3]:grid.max[dim3]: complex_z]

        my_V = V[tuple(idx)]

        if (V > 0.0).all() or (V < 0.0).all():
            print("Implicit surface will not be shown since all values have the same sign ")
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


def plot_trajectory(grid, V_all_t, trajectory):
    """
    Displays an animation of the position of a system over time

    Animation code is adapted from: https://github.com/StanfordASL/hj_reachability/blob/main/examples/quickstart.ipynb

    Args:
        grid:
        V_all_t: Value function with the last dimension being time
        trajectory: State of dynamical system. Must be same length as V_all_t.shape[-1]

    Returns:
        None
    """

    v_min, v_max = V_all_t.min(), V_all_t.max()
    levels = np.linspace(round(v_min), round(v_max), round(v_max) - round(v_min) + 1)

    dim1, dim2, dim3 = (0, 1, 2)
    complex_x = complex(0, grid.pts_each_dim[dim1])
    complex_y = complex(0, grid.pts_each_dim[dim2])
    mg_X, mg_Y = np.mgrid[grid.min[dim1]:grid.max[dim1]: complex_x, grid.min[dim2]:grid.max[dim2]:complex_y]

    fig = plt.figure(figsize=(13, 8))
    ax = plt.axes()
    plt.jet()

    point, = plt.plot(trajectory[0][0], trajectory[0][1], marker='o', color='k')

    # animation code adapted from: https://github.com/StanfordASL/hj_reachability/blob/main/examples/quickstart.ipynb
    def init():
        cf = plt.contourf(mg_X,
                          mg_Y,
                          V_all_t[:, :, 0, 0],
                          vmin=v_min,
                          vmax=v_max,
                          levels=levels)

        # TODO fix bug where a new colorbar is added every loop
        # cb = plt.colorbar()
        plt.title(f"Slice at θ_rel = {-np.pi:4.3f}", fontsize=20)
        c = plt.contour(mg_X,
                        mg_Y,
                        V_all_t[:, :, 0, -1],
                        levels=0,
                        colors="black",
                        linewidths=3)
        return c, cf, point

    def animate(i):
        cf = plt.contourf(mg_X,
                          mg_Y,
                          V_all_t[:, :, 0, i],
                          vmin=v_min,
                          vmax=v_max,
                          levels=levels)
        c = plt.contour(mg_X,
                        mg_Y,
                        V_all_t[:, :, 0, -1],
                        levels=0,
                        colors="black",
                        linewidths=3)
        point.set_data(trajectory[i][0], trajectory[i][1])
        return c, cf, point

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=V_all_t.shape[-1], interval=50)
    plt.show()
