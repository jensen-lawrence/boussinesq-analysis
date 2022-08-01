# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Standard imports
import numpy as np

# Custom imports
from utils import load_ncf, save_ncf
from fields import strain

# -----------------------------------------------------------------------------
# Dissipation Functions
# -----------------------------------------------------------------------------

# Calculate dissipation from molecular viscosity
def molecular_dissipation(data_path: str, file_num: str, out_path: str,
                          nu: float, Lx: float = 2*np.pi, Ly: float = 2*np.pi,
                          Lz: float = 2*np.pi):
    """
    Calculates kinetic energy dissipation due to molecular viscosity.

    Parameters
    ----------
    data_path : str
        path to the directory containing the .ncf strain data file
    file_num : str
        the number in the file prefix, corresponding to the output number
        from boussiensq.F90
    out_path : str
        path to where the dissipation data will be saved as a .ncf file
    nu : float
        the molecular viscosity used in the simulation that produced
        the strain data
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray]
        the time and kinetic energy dissipation data calculated from simulation
        outputs; array axes are ordered (y,z,x)
    """
    # Calculate strain data
    t, S = strain(data_path, file_num, Lx=Lx, Ly=Ly, Lz=Lz)

    # Calculate SijSij from S
    SijSij = (S**2)/2

    # Calculate and save kinetic energy dissipation
    eps = 2*nu*SijSij
    save_ncf(out_path, "EPSMOL", file_num, t, eps)

    return t, eps

# Calculate dissipation from Smagorinsky viscosity
def smagorinsky_dissipation(data_path: str, file_num: str, out_path: str,
                            n: float, Cs: float, Lx: float = 2*np.pi,
                            Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates kinetic energy dissipation due to Smagorinsky viscosity in LES.

    Parameters
    ----------
    data_path : str
        path to the directory containing the .ncf strain data file
    file_num : str
        the number in the file prefix, corresponding to the output number
        from boussiensq.F90
    out_path : str
        path to where the dissipation data will be saved as a .ncf file
    n : float
        the number of spatial grid points used in the simulation that produced
        the strain data
    Cs : float
        the Smagorinsky viscosity coefficient used in the simulation that
        produced the strain data
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray]
        the time and kinetic energy dissipation data calculated from simulation
        outputs; array axes are ordered (y,z,x)
    """
    # Calculate effective spatial grid size
    dx = 1.5*Lx/n

    # Load strain data
    t, S = strain(data_path, file_num, Lx=Lx, Ly=Ly, Lz=Lz)

    # Calculate SijSij from S
    SijSij = (S**2)/2

    # Calculate Smagorinsky viscosity
    K = (Cs*dx)**2 * S

    # Calculate and save kinetic energy dissipation
    eps = 2*K*SijSij
    save_ncf(out_path, "EPSSMAG", file_num, t, eps)

    return t, eps

# Calculate dissipation from Leith backscatter
def leith_dissipation(data_path: str, file_num: str, out_path: str):
    """
    Calculates kinetic energy dissipation due to Leith stochastic
    backscatter in LES.

    Parameters
    ----------
    data_path : str
        path to the directory containing the .ncf velocity and stochastic
        acceleration files
    file_num : str
        the number in the file prefix, corresponding to the output number
        from boussiensq.F90
    out_path : str
        path to where the dissipation data will be saved as a .ncf file

    Returns
    -------
    tuple[float, np.ndarray]
        the time and kinetic energy dissipation data calculated from simulation
        outputs; array axes are ordered (y,z,x)
    """
    # Load velocity and stochastic acceleration data
    _, u = load_ncf(data_path, "U", file_num)
    _, v = load_ncf(data_path, "V", file_num)
    _, w = load_ncf(data_path, "W", file_num)
    _, ax = load_ncf(data_path, "AX", file_num)
    _, ay = load_ncf(data_path, "AY", file_num)
    t, az = load_ncf(data_path, "AZ", file_num)

    # Calculate and save kinetic energy dissipation
    eps = u*ax + v*ay + w*az
    save_ncf(out_path, "EPSLEITH", file_num, t, eps)

    return t, eps

# Calculate dissipation from MBJ backscatter
def mbj_dissipation(data_path: str, file_num: str, out_path: str,
                    n: float, Cs: float, Lx: float = 2*np.pi,
                    Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates kinetic energy dissipation due to MBJ stochastic
    backscatter in LES.

    Parameters
    ----------
    data_path : str
        path to the directory containing the .ncf strain data file
    file_num : str
        the number in the file prefix, corresponding to the output number
        from boussiensq.F90
    out_path : str
        path to where the dissipation data will be saved as a .ncf file
    n : float
        the number of spatial grid points used in the simulation that produced
        the strain data
    Cs : float
        the Smagorinsky viscosity coefficient used in the simulation that
        produced the strain data
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray]
        the time and kinetic energy dissipation data calculated from simulation
        outputs; array axes are ordered (y,z,x)
    """
    # Calculate effective spatial grid size
    dx = 1.5*Lx/n

    # Load strain data
    _, S = strain(data_path, file_num, Lx=Lx, Ly=Ly, Lz=Lz)

    # Calculate SijSij from S
    SijSij = (S**2)/2

    # Load stochastic backscatter data
    t, Xn = load_ncf(data_path, "XM", file_num)

    # Calculate MBJ backscatter
    K = Xn * (Cs*dx)**2 * S

    # Calculate and save kinetic energy dissipation
    eps = 2*K*SijSij
    save_ncf(out_path, "EPSMBJ", file_num, t, eps)

    return t, eps

# -----------------------------------------------------------------------------