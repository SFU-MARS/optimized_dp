
class PlotOptions:
  def __init__(self, do_plot=True, plot_type="3d_plot", plotDims=[],
               slicesCut=[], min_isosurface = 0, max_isosurface = 0):
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

