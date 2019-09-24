import os
import ctypes
from ctypes import c_int, c_double, POINTER, Structure


__doc__ = """
Python wrapper for RamanOpticalControl.c
Compile with:
gcc -O3 -shared -o RamanOpticalControl.so RamanOpticalControl.c -lm -fopenmp -fPIC -lnlopt
"""


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [
        ('real', c_double),
        ('imag', c_double)
    ]


class Parameters(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [

        ('rho_0', POINTER(c_complex)),
        ('nDIM', c_int),
        ('N_exc', c_int),
        ('time_A', POINTER(c_double)),
        ('time_R', POINTER(c_double)),
        ('timeDIM_A', c_double),
        ('timeDIM_R', c_double),
        ('timeDIM_A', c_int),
        ('timeDIM_R', c_int),
        ('field_amp_A', c_double),
        ('field_amp_R', c_double),
        ('omega_R', c_double),
        ('omega_v', c_double),
        ('omega_e', c_double),
        ('d_alpha', c_double),
        ('thread_num', c_int),
        ('prob_guess_num', c_int),
        ('spectra_lower', POINTER(c_double)),
        ('spectra_upper', POINTER(c_double)),
        ('max_iter', c_int),
        ('control_guess', POINTER(c_double)),
        ('control_lower', POINTER(c_double)),
        ('control_upper', POINTER(c_double)),
        ('guess_num', c_int),
        ('max_iter_control', c_int)
    ]


class Molecule(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('nDIM', c_int),
        ('energies', POINTER(c_double)),
        ('matrix_gamma_pd', POINTER(c_double)),
        ('matrix_gamma_dep', POINTER(c_double)),
        ('gamma_dep', c_double),
        ('frequency_A', POINTER(c_double)),
        ('freqDIM_A', c_int),
        ('rho_0', POINTER(c_complex)),
        ('mu', POINTER(c_complex)),
        ('field_A', POINTER(c_complex)),
        ('field_R', POINTER(c_complex)),
        ('rho', POINTER(c_complex)),
        ('abs_spectra', POINTER(c_double)),
        ('abs_dist', POINTER(c_double)),
        ('ref_spectra', POINTER(c_double)),
        ('Raman_levels', POINTER(c_double)),
        ('levels', POINTER(c_double)),
        ('dyn_rho_A', POINTER(c_complex)),
        ('dyn_rho_R', POINTER(c_complex)),
        ('prob', POINTER(c_double))
    ]


try:
    # Load the shared library assuming that it is in the same directory
    lib1 = ctypes.cdll.LoadLibrary(os.getcwd() + "/RamanOpticalControl.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o RamanOpticalControl.so RamanOpticalControl.c -lm -lnlopt -fopenmp -fPIC
        """
    )


lib1.CalculateSpectra.argtypes = (
    POINTER(Molecule),          # molecule mol
    POINTER(Parameters),        # parameter field_params
)
lib1.CalculateSpectra.restype = POINTER(c_complex)


lib1.CalculateControl.argtypes = (
    POINTER(Molecule),          # molecule molA
    POINTER(Molecule),          # molecule molB
    POINTER(Parameters),        # parameter field_params
)
lib1.CalculateControl.restype = POINTER(c_complex)


def CalculateSpectra(mol, params):
    return lib1.CalculateSpectra(
        mol,
        params
    )

def CalculateControl(molA, molB, params):
    return lib1.CalculateControl(
        molA,
        molB,
        params
    )
