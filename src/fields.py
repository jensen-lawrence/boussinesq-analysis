# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# Standard imports
import sys
import numpy as np 
from scipy.fft import rfftn, irfftn
from numba import njit
import matplotlib.pyplot as plt 
import matplotlib.ticker as tick
from matplotlib.animation import FuncAnimation
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Custom imports
from utils import load_ncf, save_ncf, slice_ncf

# -----------------------------------------------------------------------------
# Wavenumber Functions
# -----------------------------------------------------------------------------

# Calculate x-axis wavenumber array
def calc_kxa(Lx: float, iktx: int) -> np.ndarray:
    """
    Calculates the array of x-axis wavenumbers.

    Parameters
    ----------
    Lx : float
        length of the x-axis in real space
    iktx : int
        number of x-axis wavenumbers to compute

    Returns
    -------
    np.ndarray
        array of x-axis wavenumbers
    """
    kxa = 2*np.pi/Lx * np.array([kx for kx in range(iktx)], dtype=float)
    return kxa

# Calculate y-axis or z-axis wavenumber array
def calc_kyza(L: float, ikt: int, kt: float) -> np.ndarray:
    """
    Calculates the array of y-axis or z-axis wavenumbers.

    Parameters
    ----------
    L : float
        length of the y-axis or z-axis in real space
    ikt : int
        number of y-axis or z-axis wavenumbers to compute
    kt : float
        maximum y-axis or z-axis wavenumber

    Returns
    -------
    np.ndarray
        array of y-axis or z-axis wavenumbers
    """
    ka = np.zeros(ikt, dtype=float)
    for ik in range(ikt):
        if ik < kt:
            ka[ik] = ik
        elif ik > kt:
            ka[ik] = ik - 2*kt
    ka *= 2*np.pi/L
    return ka

# Calculate y-axis wavenumber arrays
def calc_kya(Ly: float, ikty: int, kty: float) -> np.ndarray:
    """
    Calculates the array of y-axis wavenumbers.

    Parameters
    ----------
    Ly : float
        length of the y-axis in real space
    ikty : int
        number of y-axis wavenumbers to compute
    kty : float
        maximum y-axis wavenumber

    Returns
    -------
    np.ndarray
        array of y-axis wavenumbers
    """
    return calc_kyza(Ly, ikty, kty)

# Calculate z-axis wavenumber arrays
def calc_kza(Lz: float, iktz: int, ktz: float) -> np.ndarray:
    """
    Calculates the array of z-axis wavenumbers.

    Parameters
    ----------
    Lz : float
        length of the z-axis in real space
    iktz : int
        number of z-axis wavenumbers to compute
    ktz : float
        maximum z-axis wavenumber

    Returns
    -------
    np.ndarray
        array of z-axis wavenumbers
    """
    return calc_kyza(Lz, iktz, ktz)

# Calculate wavenumber arrays from data shape
def wavenumbers(nx: int, ny: int, nz: int, Lx: float, Ly: float, Lz: float):
    """
    Calculates the number of wavenumbers and the wavenumber arrays for the
    x-axis, y-axis, and z-axis.

    Parameters
    ----------
    nx : int
        number of x-axis grid points in real space
    ny : int
        number of y-axis grid points in real space
    nz : int
        number of z-axis grid points in real space
    Lx : float
        length of the x-axis in real space
    Ly : float
        length of the y-axis in real space
    Lz : float
        length of the z-axis in real space

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]
        the x-axis, y-axis, and z-axis wavenumber arrays and number of
        wavenumbers ordered (kxa, kya, kza, iktx, ikty, iktz)
    """
    ktx, kty, ktz = nx//2, ny//2, nx//2
    iktx, ikty, iktz = ktx + 1, ny, nz
    kxa = calc_kxa(Lx, iktx)
    kya = calc_kya(Ly, ikty, kty)
    kza = calc_kza(Lz, iktz, ktz)
    return kxa, kya, kza, iktx, ikty, iktz

# -----------------------------------------------------------------------------
# Gradient of a Scalar Function
# -----------------------------------------------------------------------------

# Calculate gradient in Fourier space
@njit
def gradient_k(fk: np.ndarray, kxa: np.ndarray, kya: np.ndarray,
               kza: np.ndarray, iktx: int, ikty: int, iktz: int):
    """
    Calculates the Fourier space gradient of fk, the Fourier transform of a
    real-valued 3D scalar function.

    Parameters
    ----------
    fk : np.ndarray
        Fourier transform of a real-valued 3D scalar function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        x, y, and z components of the Fourier space divergence of fk
    """
    # Initialize Fourier space gradient arrays
    gradxk = np.empty_like(fk)
    gradyk = np.empty_like(fk)
    gradzk = np.empty_like(fk)

    # Loop over all wavenumbers and calculate gradient
    for ikz in range(iktz):
        kz = kza[ikz]
        for iky in range(ikty):
            ky = kya[iky]
            for ikx in range(iktx):
                kx = kxa[ikx]
                gradxk[iky,ikz,ikx] = 1j * kx * fk[iky,ikz,ikx]
                gradyk[iky,ikz,ikx] = 1j * ky * fk[iky,ikz,ikx]
                gradzk[iky,ikz,ikx] = 1j * kz * fk[iky,ikz,ikx]

    return gradxk, gradyk, gradzk

# Calculate gradient in real space
def gradient(fr: np.ndarray, kxa: np.ndarray, kya: np.ndarray,
             kza: np.ndarray, iktx: int, ikty: int, iktz: int):
    """
    Calculates the gradient of fr, a real-valued 3D scalar function.

    Parameters
    ----------
    fr : np.ndarray
        real-valued 3D scalar function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        x, y, and z components of the divergence of fr
    """
    # FFT to Fourier space
    # Axes are (ky,kz,kx)
    fk = rfftn(fr)

    # Compute Fourier space gradient
    gradxk, gradyk, gradzk = gradient_k(fk, kxa, kya, kza, iktx, ikty, iktz)

    # IFFT to real space
    # Axes are (y,z,x)
    shape = fr.shape
    gradxr = irfftn(gradxk, s=shape)
    gradyr = irfftn(gradyk, s=shape)
    gradzr = irfftn(gradzk, s=shape)

    return gradxr, gradyr, gradzr

# -----------------------------------------------------------------------------
# Laplacian of a Scalar Function
# -----------------------------------------------------------------------------

# Calculate scalar Laplacian in Fourier space
@njit
def scalar_laplacian_k(fk: np.ndarray, kxa: np.ndarray, kya: np.ndarray,
                       kza: np.ndarray, iktx: int, ikty: int, iktz: int):
    """
    Calculates the Fourier space Laplacian of fk, the Fourier transform of a
    real-valued 3D scalar function.

    Parameters
    ----------
    fk : np.ndarray
        Fourier transform of a real-valued 3D scalar function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    np.ndarray
        the Fourier space Laplacian of fk
    """
    # Initialize Fourier space scalar Laplacian array
    Lfk = np.empty_like(fk)

    # Loop over all wavenumbers and calculate scalar Laplacian
    for ikz in range(iktz):
        kz = kza[ikz]
        for iky in range(ikty):
            ky = kya[iky]
            for ikx in range(iktx):
                kx = kxa[ikx]
                Lfk[iky,ikz,ikx] = -(kx**2 + ky**2 + kz**2)*fk[iky,ikz,ikx]

    return Lfk

# Calculate scalar Laplacian in real space
def scalar_laplacian(fr: np.ndarray, kxa: np.ndarray, kya: np.ndarray,
                     kza: np.ndarray, iktx: int, ikty: int, iktz: int):
    """
    Calculates the Laplacian of fr, a real-valued 3D scalar function.

    Parameters
    ----------
    fr : np.ndarray
        real-valued 3D scalar function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    np.ndarray
        the Laplacian of fr
    """
    # FFT to Fourier space
    # Axes are (ky,kz,kx)
    fk = rfftn(fr)

    # Compute Fourier space Laplacian
    Lfk = scalar_laplacian_k(fk, kxa, kya, kza, iktx, ikty, iktz)

    # IFFT to real space
    # Axes are (y,z,x)
    Lfr = irfftn(Lfk, s=fr.shape)

    return Lfr

# -----------------------------------------------------------------------------
# Divergence of a Vector Function
# -----------------------------------------------------------------------------

# Calculate divergence in Fourier space
@njit
def divergence_k(xk: np.ndarray, yk: np.ndarray, zk: np.ndarray,
                 kxa: np.ndarray, kya: np.ndarray, kza: np.ndarray,
                 iktx: int, ikty: int, iktz: int):
    """
    Calculates the Fourier space divergence of (xk,yk,zk), the Fourier
    transform of a real-valued 3D vector function.

    Parameters
    ----------
    xk : np.ndarray
        Fourier transform of the x component of the real-valued
        3D vector function
    yk : np.ndarray
        Fourier transform of the y component of the real-valued
        3D vector function
    zk : np.ndarray
        Fourier transform of the z component of the real-valued
        3D vector function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    np.ndarray
        the Fourier space divergence of (xk,yk,zk)
    """
    # Initialize Fourier space divergence array
    # Axes are (y,z,x)
    divk = np.empty_like(xk)

    # Loop over all wavenumbers and calculate divergence
    for ikz in range(iktz):
        kz = kza[ikz]
        for iky in range(ikty):
            ky = kya[iky]
            for ikx in range(iktx):
                kx = kxa[ikx]
                divk[iky,ikz,ikx] = 1j * (kx*xk[iky,ikz,ikx]\
                                  + ky*yk[iky,ikz,ikx]\
                                  + kz*zk[iky,ikz,ikx])

    return divk

# Calculate divergence in real space
def divergence(xr: np.ndarray, yr: np.ndarray, zr: np.ndarray,
               kxa: np.ndarray, kya: np.ndarray, kza: np.ndarray,
               iktx: int, ikty: int, iktz: int):
    """
    Calculates the divergence of (xr,yr,zr), a real-valued 3D vector function.

    Parameters
    ----------
    xr : np.ndarray
        x component of the real-valued 3D vector function
    yr : np.ndarray
        y component of the real-valued 3D vector function
    zr : np.ndarray
        z component of the real-valued 3D vector function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    np.ndarray
        the divergence of (xr,yr,zr)
    """
    # FFT to Fourier space
    # Axes are (ky,kz,kx)
    xk = rfftn(xr)
    yk = rfftn(yr)
    zk = rfftn(zr)

    # Compute Fourier space Laplacian
    divk = divergence_k(xk, yk, zk, kxa, kya, kza, iktx, ikty, iktz)

    # IFFT to real space
    # Axes are (y,z,x)
    divr = irfftn(divk, s=xr.shape)

    return divr

# -----------------------------------------------------------------------------
# Curl of a Vector Function
# -----------------------------------------------------------------------------

# Calculate curl in Fourier space
@njit
def curl_k(xk: np.ndarray, yk: np.ndarray, zk: np.ndarray,
           kxa: np.ndarray, kya: np.ndarray, kza: np.ndarray,
           iktx: int, ikty: int, iktz: int):
    """
    Calculates the Fourier space curl of (xk,yk,zk), the Fourier
    transform of a real-valued 3D vector function.

    Parameters
    ----------
    xk : np.ndarray
        Fourier transform of the x component of the real-valued
        3D vector function
    yk : np.ndarray
        Fourier transform of the y component of the real-valued
        3D vector function
    zk : np.ndarray
        Fourier transform of the z component of the real-valued
        3D vector function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        the Fourier space curl of (xk,yk,zk)
    """
    # Initialize Fourier space curl arrays
    # Axes are (y,z,x)
    curlxk = np.empty_like(xk)
    curlyk = np.empty_like(yk)
    curlzk = np.empty_like(zk)

    # Loop over all wavenumbers and calculate divergence
    for ikz in range(iktz):
        kz = kza[ikz]
        for iky in range(ikty):
            ky = kya[iky]
            for ikx in range(iktx):
                kx = kxa[ikx]
                curlxk[iky,ikz,ikx] = 1j * (ky*zk[iky,ikz,ikx]\
                                    - kz*yk[iky,ikz,ikx])
                curlyk[iky,ikz,ikx] = 1j * (kz*xk[iky,ikz,ikx]\
                                    - kx*zk[iky,ikz,ikx])
                curlzk[iky,ikz,ikx] = 1j * (kx*yk[iky,ikz,ikx]\
                                    - ky*xk[iky,ikz,ikx])

    return curlxk, curlyk, curlzk

# Calculate curl in real space
def curl(xr: np.ndarray, yr: np.ndarray, zr: np.ndarray,
         kxa: np.ndarray, kya: np.ndarray, kza: np.ndarray,
         iktx: int, ikty: int, iktz: int):
    """
    Calculates the curl of (xr,yr,zr), a real-valued 3D vector function.

    Parameters
    ----------
    xr : np.ndarray
        x component of the real-valued 3D vector function
    yr : np.ndarray
        y component of the real-valued 3D vector function
    zr : np.ndarray
        z component of the real-valued 3D vector function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        the curl of (xr,yr,zr)
    """
    # FFT to Fourier space
    # Axes are (ky,kz,kx)
    xk = rfftn(xr)
    yk = rfftn(yr)
    zk = rfftn(zr)

    # Compute Fourier space Laplacian
    curlxk, curlyk, curlzk = curl_k(xk, yk, zk, kxa, kya, kza,
                                    iktx, ikty, iktz)

    # IFFT to real space
    # Axes are (y,z,x)
    shape = xr.shape
    curlxr = irfftn(curlxk, s=shape)
    curlyr = irfftn(curlyk, s=shape)
    curlzr = irfftn(curlzk, s=shape)

    return curlxr, curlyr, curlzr

# -----------------------------------------------------------------------------
# Symmetric Part of Gradient of Vector Function
# -----------------------------------------------------------------------------

# Calculate symmetric part of vector gradient in Fourier space
@njit
def sym_gradient_k(xk: np.ndarray, yk: np.ndarray, zk: np.ndarray,
                   kxa: np.ndarray, kya: np.ndarray, kza: np.ndarray,
                   iktx: int, ikty: int, iktz: int):
    """
    Calculates the Fourier space symmetric part of the gradient
    of (xk,yk,zk), the Fourier transform of a real-valued 3D vector function.

    Parameters
    ----------
    xk : np.ndarray
        Fourier transform of the x component of the real-valued
        3D vector function
    yk : np.ndarray
        Fourier transform of the y component of the real-valued
        3D vector function
    zk : np.ndarray
        Fourier transform of the z component of the real-valued
        3D vector function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray]
        the components of the Fourier space symmetric part of the
        gradient of (xk,yk,zk)
    """
    # Initialize Fourier space symmetric gradient arrays
    # Axes are ordered (y,z,x)
    Sxxk = np.empty_like(xk)
    Syyk = np.empty_like(xk)
    Szzk = np.empty_like(xk)
    Sxyk = np.empty_like(xk)
    Syzk = np.empty_like(xk)
    Szxk = np.empty_like(xk)

    # Loop over all wavenumbers and calculate symmetric gradient
    for ikz in range(iktz):
        kz = kza[ikz]
        for iky in range(ikty):
            ky = kya[iky]
            for ikx in range(iktx):
                kx = kxa[ikx]
                Sxxk[iky,ikz,ikx] = 1j * kx * xk[iky,ikz,ikx]
                Syyk[iky,ikz,ikx] = 1j * ky * yk[iky,ikz,ikx]
                Szzk[iky,ikz,ikx] = 1j * kz * zk[iky,ikz,ikx]
                Sxyk[iky,ikz,ikx] = 1j * 0.5 * (kx*yk[iky,ikz,ikx]\
                                  + ky*xk[iky,ikz,ikx])
                Syzk[iky,ikz,ikx] = 1j * 0.5 * (ky*zk[iky,ikz,ikx]\
                                  + kz*yk[iky,ikz,ikx])
                Szxk[iky,ikz,ikx] = 1j * 0.5 * (kz*xk[iky,ikz,ikx]\
                                  + kx*zk[iky,ikz,ikx])

    return Sxxk, Syyk, Szzk, Sxyk, Syzk, Szxk

# Calculate symmetric part of vector gradient
def sym_gradient(xr: np.ndarray, yr: np.ndarray, zr: np.ndarray,
                 kxa: np.ndarray, kya: np.ndarray, kza: np.ndarray,
                 iktx: int, ikty: int, iktz: int):
    """
    Calculates the symmetric part of the gradient of (xr,yr,zr),
    a real-valued 3D vector function.

    Parameters
    ----------
    xr : np.ndarray
        x component of the real-valued 3D vector function
    yr : np.ndarray
        y component of the real-valued 3D vector function
    zr : np.ndarray
        z component of the real-valued 3D vector function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray]
        the symmetric part of the gradient of (xr,yr,zr)
    """
    # FFT to Fourier space
    # Axes are (ky,kz,kx)
    xk = rfftn(xr)
    yk = rfftn(yr)
    zk = rfftn(zr)

    # Compute Fourier space symmetric gradient
    Sxxk, Syyk, Szzk, Sxyk, Syzk, Szxk = sym_gradient_k(xk, yk, zk, kxa, kya,
                                                        kza, iktx, ikty, iktz)

    # IFFT to real space
    # Axes are (y,z,x)
    shape = xr.shape
    Sxxr = irfftn(Sxxk, s=shape)
    Syyr = irfftn(Syyk, s=shape)
    Szzr = irfftn(Szzk, s=shape)
    Sxyr = irfftn(Sxyk, s=shape)
    Syzr = irfftn(Syzk, s=shape)
    Szxr = irfftn(Szxk, s=shape)

    return Sxxr, Syyr, Szzr, Sxyr, Syzr, Szxr

# -----------------------------------------------------------------------------
# Laplacian of a Vector Function
# -----------------------------------------------------------------------------

# Calculate scalar Laplacian in Fourier space
@njit
def vector_laplacian_k(xk: np.ndarray, yk: np.ndarray, zk: np.ndarray,
                       kxa: np.ndarray, kya: np.ndarray, kza: np.ndarray,
                       iktx: int, ikty: int, iktz: int):
    """
    Calculates the Fourier space Laplacian of (xk,yk,zk), the Fourier
    transform of a real-valued 3D vector function.

    Parameters
    ----------
    xk : np.ndarray
        Fourier transform of the x component of the real-valued
        3D vector function
    yk : np.ndarray
        Fourier transform of the y component of the real-valued
        3D vector function
    zk : np.ndarray
        Fourier transform of the z component of the real-valued
        3D vector function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        the Fourier space Laplacian of (xk,yk,zk)
    """
    # Initialize Fourier space scalar Laplacian array
    Lxk = np.empty_like(xk)
    Lyk = np.empty_like(yk)
    Lzk = np.empty_like(zk)

    # Loop over all wavenumbers and calculate scalar Laplacian
    for ikz in range(iktz):
        kz = kza[ikz]
        for iky in range(ikty):
            ky = kya[iky]
            for ikx in range(iktx):
                kx = kxa[ikx]
                k2 = kx**2 + ky**2 + kz**2
                Lxk[iky,ikz,ikx] = -k2*xk[iky,ikz,ikx]
                Lyk[iky,ikz,ikx] = -k2*yk[iky,ikz,ikx]
                Lzk[iky,ikz,ikx] = -k2*zk[iky,ikz,ikx]

    return Lxk, Lyk, Lzk

# Calculate vector Laplacian in real space
def vector_laplacian(xr: np.ndarray, yr: np.ndarray, zr: np.ndarray,
                 kxa: np.ndarray, kya: np.ndarray, kza: np.ndarray,
                 iktx: int, ikty: int, iktz: int):
    """
    Calculates the Laplacian of (xr,yr,zr), a real-valued 3D vector function.

    Parameters
    ----------
    xr : np.ndarray
        x component of the real-valued 3D vector function
    yr : np.ndarray
        y component of the real-valued 3D vector function
    zr : np.ndarray
        z component of the real-valued 3D vector function
    kxa : np.ndarray
        array of x-axis wavenumbers
    kya : np.ndarray
        array of y-axis wavenumbers
    kza : np.ndarray
        array of z-axis wavenumbers
    iktx : int
        number of x-axis wavenumbers
    ikty : int
        number of y-axis wavenumbers
    iktz : int
        number of z-axis wavenumbers

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        the Laplacian of (xr,yr,zr)
    """
    # FFT to Fourier space
    # Axes are (ky,kz,kx)
    xk = rfftn(xr)
    yk = rfftn(yr)
    zk = rfftn(zr)

    # Compute Fourier space Laplacian
    Lxk, Lyk, Lzk = vector_laplacian_k(xk, yk, zk, kxa, kya, kza,
                                       iktx, ikty, iktz)

    # IFFT to real space
    # Axes are (y,z,x)
    shape = xr.shape
    Lxr = irfftn(Lxk, s=shape)
    Lyr = irfftn(Lyk, s=shape)
    Lzr = irfftn(Lzk, s=shape)

    return Lxr, Lyr, Lzr

# -----------------------------------------------------------------------------
# Field Calculation Functions
# -----------------------------------------------------------------------------

# Calculate temperature gradient
def temperature_gradient(data_path: str, file_num: str, Lx: float = 2*np.pi,
                         Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates temperature gradient from simulation temperature outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the temperature data
    file_num : str
        the output number of the temperature data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray]
        the time and temperature gradient data calculated from simulation
        outputs; array axes are ordered (y,z,x)
    """
    # Get time and real space temperature data from simulation outputs
    # Axes are (y,z,x)
    t, T = load_ncf(data_path, "TH", file_num)

    # Get wavenumbers
    ny, nz, nx = T.shape
    kxa, kya, kza, iktx, ikty, iktz = wavenumbers(nx, ny, nz, Lx, Ly, Lz)

    # Calculate temperature gradient
    gradTx, gradTy, gradTz = gradient(T, kxa, kya, kza, iktx, ikty, iktz)

    return t, gradTx, gradTy, gradTz

# Calculate temperature Laplacian
def temperature_laplacian(data_path: str, file_num: str, Lx: float = 2*np.pi,
                          Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates temperature Laplacian from simulation temperature outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the temperature data
    file_num : str
        the output number of the temperature data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray]
        the time and temperature Laplacian data calculated from simulation
        outputs; array axes are ordered (y,z,x)
    """
    # Get time and real space temperature data from simulation outputs
    # Axes are (y,z,x)
    t, T = load_ncf(data_path, "TH", file_num)

    # Get wavenumbers
    ny, nz, nx = T.shape
    kxa, kya, kza, iktx, ikty, iktz = wavenumbers(nx, ny, nz, Lx, Ly, Lz)

    # Calculate temperature Laplacian
    LT = scalar_laplacian(T, kxa, kya, kza, iktx, ikty, iktz)

    return t, LT

# Calculate speed
def speed(data_path: str, file_num: str):
    """
    Calculates speed from simulation velocity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the velocity data
    file_num : str
        the output number of the velocity data files

    Returns
    -------
    tuple[float, np.ndarray]
        the time and speed data calculated from simulation outputs;
        array axes are ordered (y,z,x)
    """
    # Get time and real space velocity data from simulation outputs
    # Axes are (y,z,x)
    _, u = load_ncf(data_path, "U", file_num)
    _, v = load_ncf(data_path, "V", file_num)
    t, w = load_ncf(data_path, "W", file_num)

    # Calculate speed
    SP = np.sqrt(u**2 + v**2 + w**2)

    return t, SP

# Calculate kinetic energy (density)
def kinetic_energy(data_path: str, file_num: str):
    """
    Calculates kinetic energy (density) from simulation velocity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the velocity data
    file_num : str
        the output number of the velocity data files

    Returns
    -------
    tuple[float, np.ndarray]
        the time and kinetic energy data calculated from simulation outputs;
        array axes are ordered (y,z,x)
    """
    # Get time and real space velocity data from simulation outputs
    # Axes are (y,z,x)
    _, u = load_ncf(data_path, "U", file_num)
    _, v = load_ncf(data_path, "V", file_num)
    t, w = load_ncf(data_path, "W", file_num)

    # Calculate kinetic energy
    KE = 0.5*(u**2 + v**2 + w**2)

    return t, KE

# Calculate divergence of velocity
def velocity_divergence(data_path: str, file_num: str, Lx: float = 2*np.pi,
                        Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates velocity divergence from simulation velocity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the velocity data
    file_num : str
        the output number of the velocity data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray]
        the time and velocity divergence data calculated from simulation
        outputs; array axes are ordered (y,z,x)
    """
    # Get time and real space velocity data from simulation outputs
    # Axes are (y,z,x)
    _, u = load_ncf(data_path, "U", file_num)
    _, v = load_ncf(data_path, "V", file_num)
    t, w = load_ncf(data_path, "W", file_num)

    # Get wavenumbers
    ny, nz, nx = u.shape
    kxa, kya, kza, iktx, ikty, iktz = wavenumbers(nx, ny, nz, Lx, Ly, Lz)

    # Calculate velocity divergence
    divu = divergence(u, v, w, kxa, kya, kza, iktx, ikty, iktz)

    return t, divu

# Calculate Laplacian of velocity
def velocity_laplacian(data_path: str, file_num: str, Lx: float = 2*np.pi,
                       Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates velocity Laplacian from simulation velocity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the velocity data
    file_num : str
        the output number of the velocity data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray]
        the time and velocity Laplacian data calculated from simulation
        outputs; array axes are ordered (y,z,x)
    """
    # Get time and real space velocity data from simulation outputs
    # Axes are (y,z,x)
    _, u = load_ncf(data_path, "U", file_num)
    _, v = load_ncf(data_path, "V", file_num)
    t, w = load_ncf(data_path, "W", file_num)

    # Get wavenumbers
    ny, nz, nx = u.shape
    kxa, kya, kza, iktx, ikty, iktz = wavenumbers(nx, ny, nz, Lx, Ly, Lz)

    # Calculate velocity Laplacian
    Lu, Lv, Lw = vector_laplacian(u, v, w, kxa, kya, kza, iktx, ikty, iktz)

    return t, Lu, Lv, Lw

# Calculate divergence of vorticity
def vorticity_divergence(data_path: str, file_num: str, Lx: float = 2*np.pi,
                         Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates vorticity divergence from simulation vorticity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the vorticity data
    file_num : str
        the output number of the vorticity data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray]
        the time and vorticity divergence data calculated from simulation
        outputs; array axes are ordered (y,z,x)
    """
    # Get time and real space vorticity data from simulation outputs
    # Axes are (y,z,x)
    _, zx = load_ncf(data_path, "ZX", file_num)
    _, zy = load_ncf(data_path, "ZY", file_num)
    t, zz = load_ncf(data_path, "ZZ", file_num)

    # Get wavenumbers
    ny, nz, nx = zx.shape
    kxa, kya, kza, iktx, ikty, iktz = wavenumbers(nx, ny, nz, Lx, Ly, Lz)

    # Calculate vorticity divergence
    divz = divergence(zx, zy, zz, kxa, kya, kza, iktx, ikty, iktz)

    return t, divz

# Calculate Laplacian of vorticity
def vorticity_laplacian(data_path: str, file_num: str, Lx: float = 2*np.pi,
                        Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates vorticity Laplacian from simulation vorticity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the vorticity data
    file_num : str
        the output number of the vorticity data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray]
        the time and vorticity Laplacian data calculated from simulation
        outputs; array axes are ordered (y,z,x)
    """
    # Get time and real space vorticity data from simulation outputs
    # Axes are (y,z,x)
    _, zx = load_ncf(data_path, "ZX", file_num)
    _, zy = load_ncf(data_path, "ZY", file_num)
    t, zz = load_ncf(data_path, "ZZ", file_num)

    # Get wavenumbers
    ny, nz, nx = zx.shape
    kxa, kya, kza, iktx, ikty, iktz = wavenumbers(nx, ny, nz, Lx, Ly, Lz)

    # Calculate vorticity Laplacian
    Lzx, Lzy, Lzz = vector_laplacian(zx, zy, zz, kxa, kya, kza,
                                     iktx, ikty, iktz)

    return t, Lzx, Lzy, Lzz

# Calculate components of strain rate tensor
def strain_components(data_path: str, file_num: str, Lx: float = 2*np.pi,
                      Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates strain rate tensor components from simulation velocity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the velocity data
    file_num : str
        the output number of the velocity data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray, np.ndarray,
          np.ndarray, np.ndarray, np.ndarray]
        the time and strain data calculated from simulation outputs;
        array axes are ordered (y,z,x)
    """
    # Get time and real space velocity data from simulation outputs
    # Axes are (y,z,x)
    _, u = load_ncf(data_path, "U", file_num)
    _, v = load_ncf(data_path, "V", file_num)
    t, w = load_ncf(data_path, "W", file_num)

    # Get wavenumbers
    ny, nz, nx = u.shape
    kxa, kya, kza, iktx, ikty, iktz = wavenumbers(nx, ny, nz, Lx, Ly, Lz)

    # Calculate strain rate components
    Sxx, Syy, Szz, Sxy, Syz, Szx = sym_gradient(u, v, w, kxa, kya, kza,
                                                iktx, ikty, iktz)

    return t, Sxx, Syy, Szz, Sxy, Syz, Szx

# Calculates the normal strain magnitude
def normal_strain(data_path: str, file_num: str, Lx: float = 2*np.pi,
                  Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates normal strain from simulation velocity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the velocity data
    file_num : str
        the output number of the velocity data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray]
        the time and strain data calculated from simulation outputs;
        array axes are ordered (y,z,x)
    """
    # Get time and strain data
    t, Sxx, Syy, Szz, _, _, _ = strain_components(data_path, file_num,
                                                  Lx=Lx, Ly=Ly, Lz=Lz)

    # Calculate SnijSnij
    SnijSnij = Sxx**2 + Syy**2 + Szz**2

    # Calculate magnitude of normal strain
    Sn = np.sqrt(2*SnijSnij)

    return t, Sn

# Calculates the shear strain magnitude
def shear_strain(data_path: str, file_num: str, Lx: float = 2*np.pi,
                 Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates shear strain from simulation velocity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the velocity data
    file_num : str
        the output number of the velocity data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray]
        the time and strain data calculated from simulation outputs;
        array axes are ordered (y,z,x)
    """
    # Get time and strain data
    t, _, _, _, Sxy, Syz, Szx = strain_components(data_path, file_num,
                                                  Lx=Lx, Ly=Ly, Lz=Lz)

    # Calculate SsijSsij
    SsijSsij = 2*Sxy**2 + 2*Syz**2 + 2*Szx**2

    # Calculate magnitude of normal strain
    Ss = np.sqrt(2*SsijSsij)

    return t, Ss

# Calculate strain rate tensor (symmetric part of velocity gradient) magnitude
def strain(data_path: str, file_num: str, Lx: float = 2*np.pi,
           Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates strain rate tensor magnitude from simulation velocity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the velocity data
    file_num : str
        the output number of the velocity data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray]
        the time and strain data calculated from simulation outputs;
        array axes are ordered (y,z,x)
    """
    # Get time and strain data
    t, Sxx, Syy, Szz, Sxy, Syz, Szx = strain_components(data_path, file_num,
                                                        Lx=Lx, Ly=Ly, Lz=Lz)

    # Calculate SijSij
    SijSij = Sxx**2 + Syy**2 + Szz**2 + 2*Sxy**2 + 2*Syz**2 + 2*Szx**2

    # Calculate magnitude of strain
    S = np.sqrt(2*SijSij)

    return t, S

# Calculate spin tensor (antisymmetric part of velocity gradient) magnitude
def spin(data_path: str, file_num: str):
    """
    Calculates spin tensor magnitude from simulation vorticity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the vorticity data
    file_num : str
        the output number of the vorticity data files

    Returns
    -------
    tuple[float, np.ndarray]
        the time and spin data calculated from simulation outputs;
        array axes are ordered (y,z,x)
    """
    # Get time and real space vorticity data from simulation outputs
    # Axes are (y,z,x)
    _, zx = load_ncf(data_path, "ZX", file_num)
    _, zy = load_ncf(data_path, "ZY", file_num)
    t, zz = load_ncf(data_path, "ZZ", file_num)

    # Calculate R
    R = np.sqrt(zx**2 + zy**2 + zz**2)

    return t, R

# Calculate Okubo-Weiss parameter
def okubo_weiss(data_path: str, file_num: str, Lx: float = 2*np.pi,
                Ly: float = 2*np.pi, Lz: float = 2*np.pi):
    """
    Calculates Okubo-Weiss parameter from simulation velocity outputs.

    Parameters
    ----------
    data_path : str
        path to the .ncf files containing the velocity and vorticity data
    file_num : str
        the output number of the velocity data files
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    tuple[float, np.ndarray]
        the time and Okubo-Weiss data calculated from simulation outputs;
        array axes are ordered (y,z,x)
    """
    # Get strain data
    _, S = strain(data_path, file_num, Lx=Lx, Ly=Ly, Lz=Lz)

    # Get spin data
    t, R = spin(data_path, file_num)

    # Calculate Okubo-Weiss parameter
    OW = S**2 - R**2

    return t, OW

# -----------------------------------------------------------------------------
# General Field Calculation
# -----------------------------------------------------------------------------

# Main function for calculating field post-simulation
def calculate_field(field: str, data_path: str, file_nums,
                    out_path: str, Lx: float = 2*np.pi, Ly: float = 2*np.pi,
                    Lz: float = 2*np.pi):
    """
    Calculates and saves the specified field.

    Parameters
    ----------
    field : str
        name of the field to be calculated; options are
        - `"temperature gradient"` / `"GRADTH"`
        - `"temperature Laplacian"` / `"LTH"`
        - `"speed"` / `"SP"`
        - `"kinetic energy"` / `"KE"`
        - `"velocity divergence"` / `"DIVU"`
        - `"velocity Laplacian`" / `"LU"`
        - `"vorticity divergence`" / `"DIVZ"`
        - `"vorticity Laplacian`" / `"LZ"`
        - `"strain components"` / `"SIJ"`
        - `"normal strain"` / `"SN"`
        - `"shear strain"` / `"SS"`
        - `"strain"` / `"S"`
        - `"spin"` / `"R"`
        - `"Okubo-Weiss"` / `"OW"`

    data_path : str
        path to the .ncf files containing the simulation data
    file_nums : str or array_like
        the output number or an array of output numbers
        of the simulation data file(s)
    out_path : str
        path to where the field data will be saved as a .ncf file
    Lx : float, optional
        length of the x-axis; set to `2*np.pi` by default
    Ly : float, optional
        length of the y-axis; set to `2*np.pi` by default
    Lz : float, optional
        length of the z-axis; set to `2*np.pi` by default

    Returns
    -------
    None
        returns None but saves field data to .ncf file(s)
    """
    # Set single file number to array for compatibility with rest of function
    if isinstance(file_nums, str):
        file_nums = np.array([file_nums])

    # Loop over file numbers and calculate fields
    for file_num in file_nums:

        # Calculate and save temperature gradient
        if (field == "temperature gradient") or (field == "GRADTH"):
            X = temperature_gradient(data_path, file_num, Lx=Lx, Ly=Ly, Lz=Lz)
            t, gradTx, gradTy, gradTz = X
            save_ncf(out_path, "GRADTHX", file_num, t, gradTx)
            save_ncf(out_path, "GRADTHY", file_num, t, gradTy)
            save_ncf(out_path, "GRADTHZ", file_num, t, gradTz)

        # Calculate and save temperature Laplacian
        elif (field == "temperature Laplacian") or (field == "LTH"):
            t, LT = temperature_laplacian(data_path, file_num,
                                           Lx=Lx, Ly=Ly, Lz=Lz)
            save_ncf(out_path, "LTH", file_num, t, LT)

        # Calculate and save speed
        elif (field == "speed") or (field == "SP"):
            t, SP = speed(data_path, file_num)
            save_ncf(out_path, "SP", file_num, t, SP)

        # Calculate and save kinetic energy (density)
        elif (field == "kinetic energy") or (field == "KE"):
            t, KE = kinetic_energy(data_path, file_num)
            save_ncf(out_path, "KE", file_num, t, KE)

        # Calculate and save velocity divergence
        elif (field == "velocity divergence") or (field == "DIVU"):
            t, divu = velocity_divergence(data_path, file_num,
                                          Lx=Lx, Ly=Ly, Lz=Lz)
            save_ncf(out_path, "DIVU", file_num, t, divu)

        # Calculate and save velocity Laplacian
        elif (field == "velocity Laplacian") or (field == "LU"):
            t, Lu, Lv, Lw = velocity_laplacian(data_path, file_num,
                                               Lx=Lx, Ly=Ly, Lz=Lz)
            save_ncf(out_path, "LU", file_num, t, Lu)
            save_ncf(out_path, "LV", file_num, t, Lv)
            save_ncf(out_path, "LW", file_num, t, Lw)

        # Calculate and save vorticity divergence
        elif (field == "vorticity divergence") or (field == "DIVZ"):
            t, divz = vorticity_divergence(data_path, file_num,
                                           Lx=Lx, Ly=Ly, Lz=Lz)
            save_ncf(out_path, "DIVZ", file_num, t, divz)

        # Calculate and save vorticity Laplacian
        elif (field == "vorticity Laplacian") or (field == "LZ"):
            t, Lzx, Lzy, Lzz = vorticity_laplacian(data_path, file_num,
                                                   Lx=Lx, Ly=Ly, Lz=Lz)
            save_ncf(out_path, "LZX", file_num, t, Lzx)
            save_ncf(out_path, "LZY", file_num, t, Lzy)
            save_ncf(out_path, "LZZ", file_num, t, Lzz)

        # Calculate and save strain components
        elif (field == "strain components") or (field == "SIJ"):
            X = strain_components(data_path, file_num, Lx=Lx, Ly=Ly, Lz=Lz)
            t, Sxx, Syy, Szz, Sxy, Syz, Szx = X
            save_ncf(out_path, "SXX", file_num, t, Sxx)
            save_ncf(out_path, "SYY", file_num, t, Syy)
            save_ncf(out_path, "SZZ", file_num, t, Szz)
            save_ncf(out_path, "SXY", file_num, t, Sxy)
            save_ncf(out_path, "SYZ", file_num, t, Syz)
            save_ncf(out_path, "SZX", file_num, t, Szx)

        # Calculate and save normal strain
        elif (field == "normal strain") or (field == "SN"):
            t, Sn = normal_strain(data_path, file_num, Lx=Lx, Ly=Ly, Lz=Lz)
            save_ncf(out_path, "SN", file_num, t, Sn)

        # Calculate and save normal strain
        elif (field == "shear strain") or (field == "SS"):
            t, Ss = shear_strain(data_path, file_num, Lx=Lx, Ly=Ly, Lz=Lz)
            save_ncf(out_path, "SS", file_num, t, Ss)

        # Calculate and save strain
        elif (field == "strain") or (field == "S"):
            t, S = strain(data_path, file_num, Lx=Lx, Ly=Ly, Lz=Lz)
            save_ncf(out_path, "S", file_num, t, S)

        # Calculate and save spin
        elif (field == "spin") or (field == "R"):
            t, R = spin(data_path, file_num)
            save_ncf(out_path, "R", file_num, t, R)

        # Calculate and save Okubo-Weiss parameter
        elif (field == "Okubo-Weiss") or (field == "OW"):
            t, OW = okubo_weiss(data_path, file_num, Lx=Lx, Ly=Ly, Lz=Lz)
            save_ncf(out_path, "OW", file_num, t, OW)

        # Quit if no or invalid field is selected
        else:
            sys.exit("Invalid field selected")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

# Heat plots of field data
def plot_field(field: str, data_path: str, file_nums, out_path: str,
               xy_idx: int, xz_idx: int, yz_idx: int, hscale: int = 8,
               vscale: int = 6, dpi: int = 200, cmap = "viridis",
               centre_cmap: bool = False, contour_levels: int = 64):
    """
    Constructs heat plots from the specified fields.

    Parameters
    ----------
    field : str
        name of the field to be plotted
    data_path : str
        path to the .ncf file(s)
    file_nums : str or array_like
        the output number or an array of output numbers
        of the simulation data file(s)
    out_path : str
        path to where the plot will be saved
    xy_idx : int
        z-coordinate index at which a slice of the xy-plane will be taken;
        no slice is taken if set to `None`
    xz_idx : int
        y-coordinate index at which a slice of the xz-plane will be taken;
        no slice is taken if set to `None`
    yz_idx : int
        x-coordinate index at which a slice of the yz-plane will be taken;
        no slice is taken if set to `None`
    hscale : int, optional
        horizontal size of the plot is determined by hscale*ncols;
        set to `8` by default
    vscale : int, optional
        vertical size of the plot is determined by vscale*nrows;
        set to `6` by default
    dpi : int, optional
        dots per inch (resolution) of the plot; set to `200` by default
    cmap : matplotlib colourmap, optional
        colourmap used in the plot; set to `"viridis"` by default
    centre_cmap : bool, optional
        centres the colourmap at 0 if `True`; set to `False` by default
    contour_levels : int, optional
        number of contour levels used in the plot; set to `64` by default

    Returns
    -------
    None
        returns None but saves the generated plot at out_path
    """
    # Set single file number to array for compatibility with rest of function
    if isinstance(file_nums, str):
        file_nums = np.array([file_nums])

    # Load time and space data
    times, data = [], []
    for file_num in file_nums:
        t, X = slice_ncf(data_path, field, file_num, xy_idx, xz_idx, yz_idx)
        times.append(t)
        data.append(X)
    times = np.array(times)
    data = np.array(data)

    # Dimensions of plot
    m = 1 if isinstance(file_nums, str) else len(file_nums)
    n = len([i for i in [xy_idx, xz_idx, yz_idx] if i is not None])

    # Get indices of plot columns and axis values
    col_idxs = {}
    if xy_idx is not None:
        col_idxs["xy"] = 0
        Lx, Ly = np.shape(data[0,col_idxs["xy"]])
        xvals = np.linspace(0, 1, Lx)
        yvals = np.linspace(0, 1, Ly)

    if xz_idx is not None:
        col_idxs["xz"] = 0 if xy_idx is None else 1
        Lx, Lz = np.shape(data[0,col_idxs["xz"]])
        xvals = np.linspace(0, 1, Lx)
        zvals = np.linspace(0, 1, Lz)

    if yz_idx is not None:
        if xz_idx is not None:
            col_idxs["yz"] = 1 if xy_idx is None else 2
        else:
            col_idxs["yz"] = 0 if xy_idx is None else 1
        Ly, Lz = np.shape(data[0,col_idxs["yz"]])
        yvals = np.linspace(0, 1, Ly)
        zvals = np.linspace(0, 1, Lz)

    # Get index of column with titles
    title_col = 1 if n == 3 else 0
        
    # Initialize plot
    fig, axes = plt.subplots(
        nrows = m,
        ncols = n,
        figsize = (int(hscale*n),int(vscale*m)),
        tight_layout = True
    )

    # Reshape one-dimensional axis array
    if (m == 1) and (n == 1):
        axes = np.array([[axes]])
    elif m == 1:
        axes = axes.reshape(1,n)
    elif n == 1:
        axes = axes.reshape(m,1)

    # Loop over plot rows and columns
    for i in range(m):
        for j in range(n):
            # Get axis and data
            ax = axes[i,j]
            d = data[i,j]

            # Set colour bar min and max
            if centre_cmap:
                datamax = np.amax(np.abs(d))
                vmin = -datamax
                vmax = datamax
            else:
                vmin = vmax = None

            # Generate the i,j plot
            if (xy_idx is not None) and (j == col_idxs["xy"]):
                cf = ax.contourf(xvals, yvals, d, contour_levels,
                                 cmap=cmap, origin="lower",
                                 vmin=vmin, vmax=vmax)
            elif (xz_idx is not None) and (j == col_idxs["xz"]):
                cf = ax.contourf(xvals, zvals, d, contour_levels,
                                 cmap=cmap, origin="lower",
                                 vmin=vmin, vmax=vmax)
            elif (yz_idx is not None) and (j == col_idxs["yz"]):
                cf = ax.contourf(yvals, zvals, d, contour_levels,
                                 cmap=cmap, origin="lower",
                                 vmin=vmin, vmax=vmax)

            # Set axis and colourbar properties
            ax.set_aspect(1.0/ax.get_data_ratio())
            cbar = fig.colorbar(cf, ax=ax)
            cbar.ax.tick_params(labelsize=18)
            if i != m-1:
                ax.tick_params(axis="x", which="both", bottom=False,
                               top=False, labelbottom=False)
                ax.tick_params(axis="y", which="both", left=False,
                               right=False, labelleft=False)

            # Set plot title
            if j == title_col:
                title = f"$t = {{{times[i]:.2f}}}$"
                ax.set_title(title, fontsize=22)

    # Add axis labels to bottom row of plots
    if xy_idx is not None:
        axes[m-1,col_idxs["xy"]].set_xlabel(r"$x/L_x$", fontsize=22)
        axes[m-1,col_idxs["xy"]].set_ylabel(r"$y/L_y$", fontsize=22)
        axes[m-1,col_idxs["xy"]].tick_params(axis='both', labelsize=18)

    if xz_idx is not None:
        axes[m-1,col_idxs["xz"]].set_xlabel(r"$x/L_x$", fontsize=22)
        axes[m-1,col_idxs["xz"]].set_ylabel(r"$z/L_z$", fontsize=22)
        axes[m-1,col_idxs["xz"]].tick_params(axis='both', labelsize=18)

    if yz_idx is not None:
        axes[m-1,col_idxs["yz"]].set_xlabel(r"$y/L_y$", fontsize=22)
        axes[m-1,col_idxs["yz"]].set_ylabel(r"$z/L_z$", fontsize=22)
        axes[m-1,col_idxs["yz"]].tick_params(axis='both', labelsize=18)

    # Save and close plot
    num_str = ""
    for file_num in file_nums:
        num_str += file_num

    idx_str = f"{xy_idx}_{xz_idx}_{yz_idx}"

    plt.savefig(f"{out_path}/{field}.{num_str}.{idx_str}.png", dpi=dpi)
    plt.close()

# -----------------------------------------------------------------------------
# Animations
# -----------------------------------------------------------------------------

# Animate field in space at a fixed time
def animate_field_space(field: str, data_path: str, file_num: str,
                        out_path: str, axes, hscale: int = 8, vscale: int = 6,
                        thinning: int = 5, dpi: int = 100, fps: int = 10,
                        cmap = "viridis", centre_cmap: bool = False,
                        contour_levels: int = 64):
    """
    Animates the specified field at a fixed time by taking slices along
    the specified axes.

    Parameters
    ----------
    field : str
        name of the field to be plotted
    data_path : str
        path to the .ncf file
    file_num : str or array_like
        the output number of the simulation data file
    out_path : str
        path to where the plot will be saved
    axes : tuple[str, str]
        set of two axes to sliced along for the animation
    hscale : int, optional
        horizontal size of the plot; set to `8` by default
    vscale : int, optional
        vertical size of the plot; set to `6` by default
    thinning : int, optional
        thins the data by the specified amount so there are fewer frames
        to animate; set to `5` by default
    dpi : int, optional
        dots per inch (resolution) of the plot; set to `100` by default
    fps : int, optional
        frames per second of the animation; set to `10` by default
    cmap : matplotlib colourmap, optional
        colourmap used in the plot; set to `"viridis"` by default
    centre_cmap : bool, optional
        centres the colourmap at 0 if `True`; set to `False` by default
    contour_levels : int, optional
        number of contour levels used in the plot; set to `64` by default

    Returns
    -------
    None
        returns None but saves the generated animation at out_path
    """
    # Load data
    _, data = load_ncf(data_path, field, file_num)
    ny, nz, nx = data.shape

    # Get plotting axes, labels, and array according to selected axes
    if axes == ("x", "y"):
        anim_arr = np.array([data[:,z,:] for z in range(nz)])
        ax_vals = (np.linspace(0, 1, nx), np.linspace(0, 1, ny))
        labels = (r"$x/L_x$", r"$y/L_y$")

    elif axes == ("y", "x"):
        anim_arr = np.array([np.transpose(data[:,z,:]) for z in range(nz)])
        ax_vals = (np.linspace(0, 1, ny), np.linspace(0, 1, nx))
        labels = (r"$y/L_y$", r"$x/L_x$")

    elif axes == ("x", "z"):
        anim_arr = np.array([data[y,:,:] for y in range(ny)])
        ax_vals = (np.linspace(0, 1, nx), np.linspace(0, 1, nz))
        labels = (r"$x/L_x$", r"$z/L_z$")

    elif axes == ("z", "x"):
        anim_arr = np.array([np.transpose(data[y,:,:]) for y in range(ny)])
        ax_vals = (np.linspace(0, 1, nz), np.linspace(0, 1, nx))
        labels = (r"$z/L_z$", r"$x/L_x$")

    elif axes == ("y", "z"):
        anim_arr = np.array([np.transpose(data[:,:,x]) for x in range(nx)])
        ax_vals = (np.linspace(0, 1, ny), np.linspace(0, 1, nz))
        labels = (r"$y/L_y$", r"$z/L_z$")

    elif axes == ("z", "y"):
        anim_arr = np.array([data[:,:,x] for x in range(nx)])
        ax_vals = (np.linspace(0, 1, nz), np.linspace(0, 1, ny))
        labels = (r"$z/L_z$", r"$y/L_y$")

    # Get min and max values
    datamin = anim_arr.min()
    datamax = anim_arr.max()
    datamaxabs = max(abs(datamin), abs(datamax))

    if centre_cmap:
        vmin = -datamaxabs
        vmax = datamaxabs
    else:
        vmin = datamin
        vmax = datamax

    # Thin animation array
    anim_arr = anim_arr[::thinning]

    # Initialize plot
    fig, ax = plt.subplots(figsize=(hscale,vscale))
    ax.set_aspect(1.0/ax.get_data_ratio())
    cf = ax.contourf(ax_vals[0], ax_vals[1], anim_arr[0], contour_levels,
                     cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cf, format=tick.FormatStrFormatter("%.2f"))
    cbar.ax.tick_params(labelsize=18)

    # Animation update function
    def update(i):
        ax.clear()
        ax.contourf(ax_vals[0], ax_vals[1], anim_arr[i], contour_levels,
                    cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xlabel(labels[0], fontsize=22)
        ax.set_ylabel(labels[1], fontsize=22)
        ax.tick_params(axis='both', labelsize=18)

    # Generate animation
    anim = FuncAnimation(fig, update, frames=range(len(anim_arr)), interval=50)
    saveas = f"{out_path}/{field}.{file_num}.{axes[0]}{axes[1]}.gif"
    anim.save(saveas, dpi=dpi, writer="pillow", fps=fps)
    plt.close()


# Animate field in time at fixed spatial slices
def animate_field_time(field: str, data_path: str, file_nums, out_path: str,
                       xy_idx: int, xz_idx: int, yz_idx: int, hscale: int = 8,
                       vscale: int = 6, dpi: int = 100, fps: int = 5,
                       cmap = "viridis", centre_cmap: bool = False,
                       contour_levels: int = 64):
    """
    Animates the specified field through time at fixed spatial slices.

    Parameters
    ----------
    field : str
        name of the field to be plotted
    data_path : str
        path to the .ncf file
    file_nums : str or array_like
        the output number or an array of output numbers
        of the simulation data file(s)
    out_path : str
        path to where the plot will be saved
    xy_idx : int
        z-coordinate index at which a slice of the xy-plane will be taken;
        no slice is taken if set to `None`
    xz_idx : int
        y-coordinate index at which a slice of the xz-plane will be taken;
        no slice is taken if set to `None`
    yz_idx : int
        x-coordinate index at which a slice of the yz-plane will be taken;
        no slice is taken if set to `None`
    hscale : int, optional
        horizontal size of the plot; set to `8` by default
    vscale : int, optional
        vertical size of the plot; set to `6` by default
    dpi : int, optional
        dots per inch (resolution) of the plot; set to `100` by default
    fps : int, optional
        frames per second of the animation; set to `10` by default
    cmap : matplotlib colourmap, optional
        colourmap used in the plot; set to `"viridis"` by default
    centre_cmap : bool, optional
        centres the colourmap at 0 if `True`; set to `False` by default
    contour_levels : int, optional
        number of contour levels used in the plot; set to `64` by default

    Returns
    -------
    None
        returns None but saves the generated animation at out_path
    """
    # Set single file number to array for compatibility with rest of function
    if isinstance(file_nums, str):
        file_nums = np.array([file_nums])

    # Load time and space data
    times, data = [], []
    for file_num in file_nums:
        t, X = slice_ncf(data_path, field, file_num, xy_idx, xz_idx, yz_idx)
        times.append(t)
        data.append(X)
    times = np.array(times)
    data = np.array(data)

    # Dimensions of plot
    m = 1 if isinstance(file_nums, str) else len(file_nums)
    n = len([i for i in [xy_idx, xz_idx, yz_idx] if i is not None])

    # Get indices of plot columns and axis values
    col_idxs = {}
    if xy_idx is not None:
        col_idxs["xy"] = 0
        Lx, Ly = np.shape(data[0,col_idxs["xy"]])
        xvals = np.linspace(0, 1, Lx)
        yvals = np.linspace(0, 1, Ly)

    if xz_idx is not None:
        col_idxs["xz"] = 0 if xy_idx is None else 1
        Lx, Lz = np.shape(data[0,col_idxs["xz"]])
        xvals = np.linspace(0, 1, Lx)
        zvals = np.linspace(0, 1, Lz)

    if yz_idx is not None:
        if xz_idx is not None:
            col_idxs["yz"] = 1 if xy_idx is None else 2
        else:
            col_idxs["yz"] = 0 if xy_idx is None else 1
        Ly, Lz = np.shape(data[0,col_idxs["yz"]])
        yvals = np.linspace(0, 1, Ly)
        zvals = np.linspace(0, 1, Lz)
        
    # Initialize plot
    fig, axes = plt.subplots(
        nrows = 1,
        ncols = n,
        figsize = (int(hscale*n),vscale),
        constrained_layout = True
    )
    divs = []

    # Reshaping one-dimensional arrays 
    if n == 1:
        divs.append(make_axes_locatable(axes))
        axes = np.array([axes])

    # Add axis labels
    if xy_idx is not None:
        divs.append(make_axes_locatable(axes[col_idxs["xy"]]))

    if xz_idx is not None:
        divs.append(make_axes_locatable(axes[col_idxs["xz"]]))

    if yz_idx is not None:
        divs.append(make_axes_locatable(axes[col_idxs["yz"]]))

    # Set colour bar axes
    divs = np.array(divs)
    caxes = []
    for div in divs:
        caxes.append(div.append_axes("right", "5%", "5%"))
    caxes = np.array(caxes)

    # Animation update function
    def update(i):
        for j in range(n):
            # Get axis and data
            ax = axes[j]
            cax = caxes[j]
            d = data[i,j]
            ax.clear()
            cax.clear()

            # Set colour bar min and max
            if centre_cmap:
                datamax = np.amax(np.abs(data))
                vmin = -datamax
                vmax = datamax
            else:
                vmin = vmax = None

            # Generate the i,j plot
            if (xy_idx is not None) and (j == col_idxs["xy"]):
                cf = ax.contourf(xvals, yvals, d, contour_levels,
                            cmap=cmap, origin="lower",
                            vmin=vmin, vmax=vmax)
                ax.set_xlabel(r"$x/L_x$", fontsize=22)
                ax.set_ylabel(r"$y/L_y$", fontsize=22)
                ax.tick_params(axis='both', labelsize=18)
            elif (xz_idx is not None) and (j == col_idxs["xz"]):
                cf = ax.contourf(xvals, zvals, d, contour_levels,
                            cmap=cmap, origin="lower",
                            vmin=vmin, vmax=vmax)
                ax.set_xlabel(r"$x/L_x$", fontsize=22)
                ax.set_ylabel(r"$z/L_z$", fontsize=22)
                ax.tick_params(axis='both', labelsize=18)
            elif (yz_idx is not None) and (j == col_idxs["yz"]):
                cf = ax.contourf(yvals, zvals, d, contour_levels,
                            cmap=cmap, origin="lower",
                            vmin=vmin, vmax=vmax)
                ax.set_xlabel(r"$y/L_y$", fontsize=22)
                ax.set_ylabel(r"$z/L_z$", fontsize=22)
                ax.tick_params(axis='both', labelsize=18)

            # Set colourbar properties
            fig.colorbar(cf, cax=cax, format=tick.FormatStrFormatter("%.2f"))
            cax.tick_params(labelsize=18)

    # Animate and save plot
    anim = FuncAnimation(fig, update, frames=np.arange(m), interval=50)
    saveas = f"{out_path}/{field}.gif"
    anim.save(saveas, dpi=dpi, writer="pillow", fps=fps)
    plt.close()

# -----------------------------------------------------------------------------