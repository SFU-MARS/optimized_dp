class PlotOptions:
    def __init__(self, plot_type="3d_plot", dims_plot=[], slices=[]):
        if plot_type not in ["2d_plot", "3d_plot"]:
            raise Exception("Illegal plot type !")

        if plot_type == "2d_plot":
            if len(dims_plot) != 2:
                raise Exception("Make sure that dim_plot size is 2 !!")

        if plot_type == "3d_plot":
            if len(dims_plot) != 3:
                raise Exception("Make sure that dim_plot size is 3 !!")

        self.dims_plot = dims_plot
        self.plot_type = plot_type
        self.slices = slices
