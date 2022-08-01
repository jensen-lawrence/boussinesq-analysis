# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Standard imports
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.cm import get_cmap

# Custom imports
from constants import eng_vars_dict, eng_labels_arr
from constants import eps_vars_dict, eps_labels_arr
from utils import get_file_name

# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------

# Get time series data
def time_data(file: str, vars, vars_dict: dict, labels_arr):
    """
    Gets time and variable data for the specified variable(s) from the
    specified simulation output file.

    Parameters
    ----------
    file : str
        path to the file containing the time series data; the file itself
        must be either `eng.dat` or `eps.dat`
    vars : str or array_like
        the variable(s) of interest
    vars_dict : dict
        dictionary containing the data file indices corresponding to the
        specified variable(s)
    labels_arr : array_like
        array-like object containing the plotting labels corresponding to the
        specified variable(s)

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        time data, variable(s) data, and variable label(s)
    """
    # Load data
    data = np.loadtxt(file)
    t = data[:,0]

    # Set single variable to array for compatibility with rest of function
    if isinstance(vars, str):
        vars = np.array([vars])

    # Get variable data
    idxs = np.array([vars_dict[var] for var in vars])
    labels = np.array([labels_arr[idx-1] for idx in idxs])
    y = np.array([data[:,idx] for idx in idxs])

    return t, y, labels

# Get data from eng.dat
def eng_data(file: str, vars):
    """
    Gets time and variable data for the specified variable(s) stored
    in `eng.dat`.

    Parameters
    ----------
    file : str
        path to `eng.dat`
    vars : str or array_like
        the variable(s) of interest

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        time data, variable(s) data, and variable label(s)
    """
    t, y, labels = time_data(file, vars, eng_vars_dict, eng_labels_arr)
    return t, y, labels

# Get data from eps.dat
def eps_data(file: str, vars):
    """
    Gets time and variable data for the specified variable(s) stored
    in `eps.dat`.

    Parameters
    ----------
    file : str
        path to `eps.dat`
    vars : str or array_like
        the variable(s) of interest

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        time data, variable(s) data, and variable label(s)
    """
    t, y, labels = time_data(file, vars, eps_vars_dict, eps_labels_arr)
    return t, y, labels

# -----------------------------------------------------------------------------
# PLots
# -----------------------------------------------------------------------------

# Time series plot
def plot_time_series(file: str, vars, saveas: str, hscale: int = 12,
                     vscale: int = 8, dpi: int = 300, colour = "dodgerblue",
                     cmap = "turbo", grid: bool = True):
    """
    Constructs a time series plot of the specified variable(s) from the
    specified simulation output file.

    Parameters
    ----------
    file : str
        path to the file containing the time series data; the file itself
        must be either `eng.dat` or `eps.dat`
    vars : str or array_like
        the variable(s) to be plotted
    saveas : str
        path and file name the plot will be saved to
    hscale : int, optional
        determines horizontal scale of plot; set to `12` by default
    vscale : int, optional
        determines vertical scale of the plot; set to `8` by default
    dpi : int, optional
        dots per inch (resolution) of the plot; set to `300` by default
    colour : str or tuple, optional
        colour used for plots of one variable; set to `"dodgerblue"` by default
    cmap : str or any matplotlib colourmap, optional
        colourmap sampled from for plots of multiple variables;
        set to `"turbo"` by default
    grid : bool, optional
        determines whether the plot has a grid; set to `True` by default

    Returns
    -------
    None
        no return, but saves the generated plot to the specified path
    """
    # Determine if data is from eng.dat or eps.dat
    filename = get_file_name(file)
    assert filename in ("eng", "eps"), "file must be eng.dat or eps.dat"

    # Get data
    if filename == "eng":
        t, y, labels = eng_data(file, vars)
        var_kind = r"$E$"

    elif filename == "eps":
        t, y, labels = eps_data(file, vars)
        var_kind = r"$\varepsilon$"

    # Set single variable to array for compatibility with rest of function
    if isinstance(vars, str):
        vars = np.array([vars])
    n = len(vars)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(hscale,vscale), constrained_layout=True)
    ax.grid(grid)
    ax.set_xlabel(r"$t$", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)

    # Plot variables
    if n == 1:
        ax.plot(t, y[0], lw=2, c=colour)
        ax.set_ylabel(labels[0], fontsize=22)

    else:
        crange = np.linspace(0.12, 0.88, n)
        cmap_func = get_cmap(cmap)
        for i in range(n):
            ax.plot(t, y[i], lw=2, c=cmap_func(crange[i]), label=labels[i])
        ax.set_ylabel(var_kind, fontsize=22)
        ax.legend(loc="best", fontsize=18)

    # Save plot
    plt.savefig(saveas, dpi=dpi)
    plt.close()

# -----------------------------------------------------------------------------