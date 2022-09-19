import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import Rectangle
from matplotlib import colors


def heatmap(array, c='viridis', ax=None, cbarlabel="", row_labels=None, col_labels=None, xlabel_rotation=0, datamin=None, datamax=None, logscale_cmap=False, annotate=False, annotation_color='k', title=None, xlabel=None, ylabel=None):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.

    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if ax is None:
        ax = plt.gca()
    cmap = getattr(plt.cm, c)
    
    # get array size and xy labels
    data = array.copy()
    ny,nx = data.shape
    if row_labels is None:
        row_labels = list(np.arange(ny)+1)
    if col_labels is None:
        col_labels = list(np.arange(nx)+1)

    # get locations of nan values and negative values, replace values so these don't trigger an error
    naninds = np.where(np.isnan(data) == True)
    infinds = np.where(np.isinf(data) == True)
    data[infinds] = np.nan
    data[naninds] = np.nanmean(data)
    data[infinds] = np.nanmean(data)
    data_cmap = data.copy()
        
    # get colormap to plot
    if logscale_cmap: # plot on logscale
        if datamin is None:
            datamin = np.nanmin(np.abs(data_cmap))
        if datamax is None:
            datamax = np.nanmax(np.abs(data_cmap))
        data_cmap = np.log(np.abs(data_cmap))
        datamin, datamax = np.log(datamin), np.log(datamax)
    else:
        if datamin is None:
            datamin = np.nanmin(data_cmap)
        if datamax is None:
            datamax = np.nanmax(data_cmap)
        
    # get cmap gradations
    dataint = (datamax-datamin)/100
    norm = plt.Normalize(datamin, datamax+dataint)
    # convert data array into colormap
    colormap = cmap(norm(data_cmap))
        
    # Set the positions of nan values in colormap to 'lime'
    colormap[naninds[0], naninds[1], :3] = 0,1,0
    colormap[infinds[0], infinds[1], :3] = 1,1,1

    # plot colormap
    im = ax.imshow(colormap, interpolation='nearest')

    # Create colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.07)
    cbar = ax.figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    
    if logscale_cmap == True:
        cbar_labels = cbar.ax.get_yticks()
        cbar.set_ticks(cbar_labels)
        cbar_labels_unlog = list(np.round(np.exp(np.array(cbar_labels)),2))
        cbar.set_ticklabels(cbar_labels_unlog)
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
#     ax.set_xlabel(loc='top')
    ax.set_xticklabels(col_labels, rotation=xlabel_rotation, ha='left', fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    
    # Annotate
    if annotate:
        annotate_heatmap(array, ax, ndecimals=3, c=annotation_color)

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    if title is not None:
        ax.set_title(title)
    if xlabel is not None: 
        ax.set_xlabel(xlabel)
    if ylabel is not None: 
        ax.set_ylabel(ylabel)
    plt.show()

    return im, cbar
    