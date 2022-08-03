# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Standard imports
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.cm import get_cmap
from matplotlib.animation import FuncAnimation

# Custom imports
from constants import spc_vars_dict, spc_labels_arr
from constants import trn_vars_dict, trn_labels_arr
from utils import get_file_name

# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------

# Get spectrum data
def spectrum_data(file: str, var: str, t0: float, tf: float,
                  vars_dict: dict, labels_arr):
    """
    Gets time, spectrum, and variable data for the specified variable 
    from the specified simulation output file.

    Parameters
    ----------
    file : str
        path to the file containing the spectrum data; the file itself
        must be `spc.dat`, `spch.dat`, `spcz.dat`, `trn.dat`, `trnh.dat`, 
        or `trnz.dat`
    var : str
        the variable of interest
    t0 : float
        the time at which the spectrum data starts
    tf : float
        the time at which the spectrum data ends
    vars_dict : dict
        dictionary containing the data file indices corresponding to the
        specified variable
    labels_arr : array_like
        array-like object containing the plotting labels corresponding to the
        specified variable

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, str]
        time data, spectrum data, variable data, and variable label
    """
    # Load data
    data = np.loadtxt(file)
    all_k = data[:,0]
    k = np.unique(all_k)
    t = np.linspace(t0, tf, int(all_k.size/k.size))

    # Cut off initial time if data is trn
    filename = get_file_name(file)
    if filename == "trn":
        t = t[1:]

    # Get variable data
    idx = vars_dict[var]
    label = labels_arr[idx-1]
    y = data[:,idx].reshape(t.size, k.size)

    return t, k, y, label

# Get data from spc(h/z).dat
def spc_data(file: str, var: str, t0: float, tf: float):
    """
    GGets time, spectrum, and variable data for the specified variable 
    from `spc.dat`, `spch.dat`, or `spcz.dat`

    Parameters
    ----------
    file : str
        path to `spc.dat`, `spch.dat`, or `spcz.dat` 
    var : str
        the variable of interest
    t0 : float
        the time at which the spectrum data starts
    tf : float
        the time at which the spectrum data ends

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, str]
        time data, spectrum data, variable data, and variable label
    """
    t, k, y, label = spectrum_data(file, var, t0, tf,
                                   spc_vars_dict, spc_labels_arr)
    return t, k, y, label

# Get data from trn(h/z).dat
def trn_data(file: str, var: str, t0: float, tf: float):
    """
    GGets time, spectrum, and variable data for the specified variable 
    from `trn.dat`, `trnh.dat`, or `trnz.dat`

    Parameters
    ----------
    file : str
        path to `trn.dat`, `trnh.dat`, or `trnz.dat` 
    var : str
        the variable of interest
    t0 : float
        the time at which the spectrum data starts
    tf : float
        the time at which the spectrum data ends

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, str]
        time data, spectrum data, variable data, and variable label
    """
    t, k, y, label = spectrum_data(file, var, t0, tf,
                                   trn_vars_dict, trn_labels_arr)
    return t, k, y, label

# -----------------------------------------------------------------------------
# Wavenumber Corrections
# -----------------------------------------------------------------------------

# Horizontal wavenumber correction
def horizontal_correction(file: str, k: np.ndarray, y: np.ndarray):
    """
    Calculates the cutoff and correction to a horizontal wavenumber spectrum.

    Parameters
    ----------
    file : str
        path to the file containing the spectrum data; the file itself
        must be `spc.dat`, `spch.dat`, `spcz.dat`, `trn.dat`, `trnh.dat`, 
        or `trnz.dat`
    k : np.ndarray
        array of horizontal wavenumbers
    y : np.ndarray
        variable data

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        corrected wavenumbers and variable data
    """
    # Get number of sampled nodes
    nmodes = np.loadtxt(file)[:,-1][0:k.size]

    # Calculate maximum horizontal wavenumber
    kt = int(max(k)/np.sqrt(2))

    # Calculate correction coefficients
    coeffs = np.array([(2*np.pi*k[i]*(2*kt + 1))/nmodes[i] for i in range(k.size)])

    # Loop over data and apply correction
    y_corr = []
    for yi in y:
        yi_corr = [coeffs[j] * yi[j] for j in range(yi.size)]
        y_corr.append(yi_corr)

    # Slice corrected data and wavenumbers
    y_corr = np.array(y_corr)[:,:kt + 1]
    k_corr = np.arange(min(k), kt + 1)

    return k_corr, y_corr

# Total wavenumber correction
def total_correction(file: str, k: np.ndarray, y: np.ndarray):
    """
    Calculates the cutoff and correction to a total wavenumber spectrum.

    Parameters
    ----------
    file : str
        path to the file containing the spectrum data; the file itself
        must be `spc.dat`, `spch.dat`, `spcz.dat`, `trn.dat`, `trnh.dat`, 
        or `trnz.dat`
    k : np.ndarray
        array of total wavenumbers
    y : np.ndarray
        variable data

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        corrected wavenumbers and variable data
    """
    # Get number of sampled nodes
    nmodes = np.loadtxt(file)[:,-1][0:k.size]

    # Calculate maximum horizontal wavenumber
    kt = int(max(k)/np.sqrt(3))

    # Calculate correction coefficients
    coeffs = np.array([(4*np.pi*k[i]**2)/nmodes[i] for i in range(k.size)])

    # Loop over data and apply correction
    y_corr = []
    for yi in y:
        yi_corr = [coeffs[j] * yi[j] for j in range(yi.size)]
        y_corr.append(yi_corr)

    # Slice corrected data and wavenumbers
    y_corr = np.array(y_corr)[:,:kt]
    k_corr = np.arange(min(k), kt + 1)

    return k_corr, y_corr

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

# Spectrum line plot at sampled times
def plot_spectrum_1D(file: str, var: str, times, saveas: str, t0: float = 0.0,
                     tf: float = 4.0, correction: bool = False,
                     ref_slope = None, xlog: bool = True, ylog: bool = True,
                     hscale: int = 12, vscale: int = 8, dpi: int = 300,
                     colour = "dodgerblue", cmap = "turbo", grid: bool = True):
    """
    Constructs a spectrum plot of the specified variable from the
    specified simulation output file at the specified time(s).

    Parameters
    ----------
    file : str
        path to the file containing the spectrum data; the file itself
        must be `spc.dat`, `spch.dat`, `spcz.dat`, `trn.dat`, `trnh.dat`, 
        or `trnz.dat`
    var : str
        the variable to be plotted
    times : float or array_like of float
        time(s) at which the variable will be plotted
    saveas : str
        path and file name the plot will be saved to
    t0 : float, optional
        the time at which the spectrum data starts; set to `0.0` by default
    tf : float, optional
        the time at which the spectrum data ends; set to `4.0` by default
    correction : bool, optional
        determines whether the wavenumber correction is applied to the data;
        set to `False` by default
    ref_slope : int, float, str, or None, optional
        adds the curve k^ref_slope to the plot if ref_slope is not None;
        set to `None` by default
    xlog : bool, optional
        scales the x-axis with a log10 scale; set to `True` by default
    ylog : bool, optional
        scales the y-axis with a log10 scale; set to `True` by default
    hscale : int, optional
        determines horizontal scale of plot; set to `12` by default
    vscale : int, optional
        determines vertical scale of the plot; set to `8` by default
    dpi : int, optional
        dots per inch (resolution) of the plot; set to `300` by default
    colour : str or tuple, optional
        colour used for plots of one time; set to `"dodgerblue"` by default
    cmap : str or any matplotlib colourmap, optional
        colourmap sampled from for plots of multiple times;
        set to `"turbo"` by default
    grid : bool, optional
        determines whether the plot has a grid; set to `True` by default

    Returns
    -------
    None
        no return, but saves the generated plot to the specified path
    """
    # Determine source of data
    filename = get_file_name(file)
    assert filename in ("spc", "spch", "spcz", "trn", "trnh", "trnz"), \
        "file must be spc(h/z).dat or trn(h/z).dat"

    # Set single time to array for compatibility
    if isinstance(times, int) or isinstance(times, float):
        times = [times]
    n = len(times)
        
    # Wavenumber labels
    if "h" in filename:
        xlabel = r"$k_h$"
    elif "z" in filename:
        xlabel = r"$k_z$"
    else:
        xlabel = r"$k$"

    # Load data
    if "spc" in filename:
        t, k, y, ylabel = spc_data(file, var, t0, tf)
    elif "trn" in filename:
        t, k, y, ylabel = trn_data(file, var, t0, tf)

    # Apply correction
    if correction:
        if "h" in filename:
            k, y = horizontal_correction(file, k, y)
        else:
            k, y = total_correction(file, k, y)

    # Get data at specified time values
    yvals = []
    for ti in times:
        t_idx = np.where(t == ti)[0][0]
        yvals.append(y[t_idx,:])
    yvals = np.array(yvals)

    # Initialize plot
    fig, ax = plt.subplots(figsize=(hscale,vscale), constrained_layout=True)
    ax.grid(grid)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)

    # Plot data at each time
    if n == 1:
        ax.plot(k, yvals[0], lw=2, c=colour, label=f"$t = {{{times[0]}}}$")

    else:
        crange = np.linspace(0.12, 0.88, n)
        cmap_func = get_cmap(cmap)
        for i in range(n):
            ax.plot(k, yvals[i], lw=2, c=cmap_func(crange[i]),
                    label=f"$t = {{{times[i]}}}$")

    # Plot reference slope
    if ref_slope is not None:
        if isinstance(ref_slope, int) or isinstance(ref_slope, float):
            ax.plot(k, k**ref_slope, lw=1, ls="-.", c="black",
                    label=f"$k^{{{ref_slope}}}$")
        elif isinstance(ref_slope, str):
            try:
                slope_val = float(ref_slope)
            except ValueError:
                if ref_slope.count("/") == 1:
                    frac = ref_slope.split("/")
                    slope_val = float(frac[0])/float(frac[1])
            ax.plot(k, k**slope_val, lw=1, ls="-.", c="black",
                    label=f"$k^{{{ref_slope}}}$")

    ax.legend(loc="best", fontsize=18)

    # Save plot
    plt.savefig(saveas, dpi=dpi)
    plt.close()

# Spectrum heat plot at all times
def plot_spectrum_2D(file: str, var: str, saveas: str, t0: float = 0.0,
                     tf: float = 4.0, correction: bool = False,
                     xlog: bool = True, hscale: int = 10, vscale: int = 8,
                     dpi: int = 300, cmap = "viridis",
                     centre_cmap: bool = False, contour_levels: int = 64):
    """
    Constructs a spectrum plot of the specified variable from the
    specified simulation output file at the specified time(s).

    Parameters
    ----------
    file : str
        path to the file containing the spectrum data; the file itself
        must be `spc.dat`, `spch.dat`, `spcz.dat`, `trn.dat`, `trnh.dat`, 
        or `trnz.dat`
    var : str
        the variable to be plotted
    saveas : str
        path and file name the plot will be saved to
    t0 : float, optional
        the time at which the spectrum data starts; set to `0.0` by default
    tf : float, optional
        the time at which the spectrum data ends; set to `4.0` by default
    correction : bool, optional
        determines whether the wavenumber correction is applied to the data;
        set to `False` by default
    xlog : bool, optional
        scales the x-axis with a log10 scale; set to `True` by default
    hscale : int, optional
        determines horizontal scale of plot; set to `10` by default
    vscale : int, optional
        determines vertical scale of the plot; set to `8` by default
    dpi : int, optional
        dots per inch (resolution) of the plot; set to `300` by default
    cmap : str or any matplotlib colourmap, optional
        colourmap sampled from for plots of multiple variables;
        set to `"viridis"` by default
    centre_cmap : bool, optional
        centres the colourmap at 0 if `True`; set to `False` by default
    contour_levels : int, optional
        number of contour levels used in the plot; set to `64` by default

    Returns
    -------
    None
        no return, but saves the generated plot to the specified path
    """
    # Determine source of data
    filename = get_file_name(file)
    assert filename in ("spc", "spch", "spcz", "trn", "trnh", "trnz"), \
        "file must be spc(h/z).dat or trn(h/z).dat"
        
    # Wavenumber labels
    if "h" in filename:
        xlabel = r"$k_h$"
    elif "z" in filename:
        xlabel = r"$k_z$"
    else:
        xlabel = r"$k$"

    # Load data
    if "spc" in filename:
        t, k, y, ylabel = spc_data(file, var, t0, tf)
    elif "trn" in filename:
        t, k, y, ylabel = trn_data(file, var, t0, tf)

    # Apply correction
    if correction:
        if "h" in filename:
            k, y = horizontal_correction(file, k, y)
        else:
            k, y = total_correction(file, k, y)

    # Set colour bar min and max
    if centre_cmap:
        datamax = np.amax(np.abs(y))
        vmin = -datamax
        vmax = datamax
    else:
        vmin = vmax = None

    # Generate plot
    fig, ax = plt.subplots(figsize=(hscale,vscale), tight_layout=True)
    CS = ax.contourf(k[1:], t[1:], y[1:,1:], contour_levels, cmap=cmap,
                     origin="lower", vmin=vmin, vmax=vmax) 
    cbar = fig.colorbar(CS)
    cbar.ax.set_ylabel(ylabel, fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(r"$t$", fontsize=22)
    ax.tick_params(axis="both", labelsize=18)

    if xlog:
        ax.set_xscale("log")

    # Save plot
    plt.savefig(saveas, dpi=dpi)
    plt.close()

# -----------------------------------------------------------------------------
# Animations
# -----------------------------------------------------------------------------

# Animated line plot of spectrum through time
def animate_spectrum(file: str, var: str, saveas: str, t0: float = 0.0,
                     tf: float = 4.0, correction: bool = False,
                     xlog: bool = True, ylog: bool = True,
                     hscale: int = 12, vscale: int = 8, thinning: int = 10,
                     dpi: int = 300, fps: int = 10, colour = "dodgerblue",
                     grid: bool = True):
    """
    Constructs a spectrum plot of the specified variable from the
    specified simulation output file at the specified time(s).

    Parameters
    ----------
    file : str
        path to the file containing the spectrum data; the file itself
        must be `spc.dat`, `spch.dat`, `spcz.dat`, `trn.dat`, `trnh.dat`, 
        or `trnz.dat`
    var : str
        the variable to be plotted
    saveas : str
        path and file name the plot will be saved to
    t0 : float, optional
        the time at which the spectrum data starts; set to `0.0` by default
    tf : float, optional
        the time at which the spectrum data ends; set to `4.0` by default
    correction : bool, optional
        determines whether the wavenumber correction is applied to the data;
        set to `False` by default
    xlog : bool, optional
        scales the x-axis with a log10 scale; set to `True` by default
    ylog : bool, optional
        scales the y-axis with a log10 scale; set to `True` by default
    hscale : int, optional
        determines horizontal scale of plot; set to `12` by default
    vscale : int, optional
        determines vertical scale of the plot; set to `8` by default
    thinning : int, optional
        thins the data by the specified amount so there are fewer frames
        to animate; set to `10` by default
    dpi : int, optional
        dots per inch (resolution) of the plot; set to `300` by default
    fps : int, optional
        framerate of the animation; set to `10` by default
    grid : bool, optional
        determines whether the plot has a grid; set to `True` by default

    Returns
    -------
    None
        no return, but saves the generated animation to the specified path
    """
    # Determine source of data
    filename = get_file_name(file)
    assert filename in ("spc", "spch", "spcz", "trn", "trnh", "trnz"), \
        "file must be spc(h/z).dat or trn(h/z).dat"
        
    # Wavenumber labels
    if "h" in filename:
        xlabel = r"$k_h$"
    elif "z" in filename:
        xlabel = r"$k_z$"
    else:
        xlabel = r"$k$"

    # Load data
    if "spc" in filename:
        t, k, y, ylabel = spc_data(file, var, t0, tf)
    elif "trn" in filename:
        t, k, y, ylabel = trn_data(file, var, t0, tf)

    # Apply correction
    if correction:
        if "h" in filename:
            k, y = horizontal_correction(file, k, y)
        else:
            k, y = total_correction(file, k, y)

    # Thin data
    t = t[::thinning]
    y = y[::thinning,:]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(hscale,vscale), tight_layout=True)
    ax.grid(grid)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=18)
    ax.set_xlabel(xlabel, fontsize=22)
    ax.set_ylabel(ylabel, fontsize=22)
    line, = ax.plot(k, y[0,:], lw=2, c=colour)

    # Animation update function
    def update(i):
        line.set_ydata(y[i,:])

    # Animate
    anim = FuncAnimation(fig, update, frames=np.arange(t.size), interval=50)
    anim.save(saveas, dpi=dpi, writer="pillow", fps=fps)
    plt.close()

# -----------------------------------------------------------------------------