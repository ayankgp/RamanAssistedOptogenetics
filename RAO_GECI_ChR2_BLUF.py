#!/usr/bin/env python

"""
RAO_GECI_ChR2_BLUF.py:

Class containing C calls for spectra calculation and Raman control.
Plots results obtained from C calls.
"""

__author__ = "Ayan Chattopadhyay"
__affiliation__ = "Princeton University"


# ---------------------------------------------------------------------------- #
#                           LOADING LIBRARY HEADERS                            #
# ---------------------------------------------------------------------------- #

import numpy as np
from types import MethodType, FunctionType
from wrapper import *
import pandas as pd
from scipy.interpolate import interp1d
from multiprocessing import cpu_count
from ctypes import c_int, c_double, c_char_p, POINTER, Structure
from matplotlib import cm


class ADict(dict):
    """
    Dictionary where you can access keys as attributes
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class RamanOpticalControl:
    """
    Main class initializing molecule and spectra calculation, Raman control
    optimization routines on the molecule.
    """

    def __init__(self, params, **kwargs):
        """
        __init__ function call to initialize variables from the
        parameters for the class instance provided in __main__ and
        add new variables for use in other functions in this class.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.time_A = np.linspace(-params.timeAMP_A, params.timeAMP_A, params.timeDIM_A)
        self.time_R = np.linspace(-params.timeAMP_R, params.timeAMP_R, params.timeDIM_R)

        self.field_A = np.zeros(params.timeDIM_A, dtype=np.complex)
        self.field_R = np.zeros(params.timeDIM_R, dtype=np.complex)

        self.matrix_gamma_pd = np.ascontiguousarray(self.matrix_gamma_pd)
        self.matrix_gamma_dep_GECI = np.ascontiguousarray(self.matrix_gamma_dep)
        self.matrix_gamma_dep_ChR2 = np.ascontiguousarray(self.matrix_gamma_dep)

        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(params.rho_0)
        self.rho_GECI = np.ascontiguousarray(params.rho_0.copy())
        self.rho_ChR2 = np.ascontiguousarray(params.rho_0.copy())
        self.energies_GECI = np.ascontiguousarray(self.energies_GECI)
        self.energies_ChR2 = np.ascontiguousarray(self.energies_ChR2)

        self.N = len(self.energies_GECI)

        self.abs_spectra_GECI = np.ascontiguousarray(np.zeros(len(self.frequency_A_GECI)))
        self.abs_spectra_ChR2 = np.ascontiguousarray(np.zeros(len(self.frequency_A_ChR2)))

        self.abs_dist_GECI = np.ascontiguousarray(np.empty((len(self.prob_GECI), len(self.frequency_A_GECI))))
        self.abs_dist_ChR2 = np.ascontiguousarray(np.empty((len(self.prob_ChR2), len(self.frequency_A_ChR2))))

        self.dyn_rho_A_GECI = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)
        self.dyn_rho_A_ChR2 = np.ascontiguousarray(np.zeros((N, params.timeDIM_A)), dtype=np.complex)

        self.dyn_rho_R_GECI = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)
        self.dyn_rho_R_ChR2 = np.ascontiguousarray(np.zeros((N, params.timeDIM_R)), dtype=np.complex)

    def create_molecules(self, GECI, ChR2):
        """
        Creates molecules from class parameters
        """
        #  ----------------------------- CREATING GECI ------------------------  #

        GECI.nDIM = len(self.energies_GECI)
        GECI.energies = self.energies_GECI.ctypes.data_as(POINTER(c_double))
        GECI.matrix_gamma_pd = self.matrix_gamma_pd.ctypes.data_as(POINTER(c_double))
        GECI.matrix_gamma_dep = self.matrix_gamma_dep_GECI.ctypes.data_as(POINTER(c_double))
        GECI.gamma_dep = self.gamma_dep_GECI
        GECI.frequency_A = self.frequency_A_GECI.ctypes.data_as(POINTER(c_double))
        GECI.freqDIM_A = len(self.frequency_A_GECI)
        GECI.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        GECI.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        GECI.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        GECI.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))

        GECI.rho = self.rho_GECI.ctypes.data_as(POINTER(c_complex))
        GECI.abs_spectra = self.abs_spectra_GECI.ctypes.data_as(POINTER(c_double))
        GECI.abs_dist = self.abs_dist_GECI.ctypes.data_as(POINTER(c_double))
        GECI.ref_spectra = self.ref_spectra_GECI.ctypes.data_as(POINTER(c_double))
        GECI.Raman_levels = self.Raman_levels_GECI.ctypes.data_as(POINTER(c_double))
        GECI.levels = self.levels_GECI.ctypes.data_as(POINTER(c_double))
        GECI.dyn_rho_A = self.dyn_rho_A_GECI.ctypes.data_as(POINTER(c_complex))
        GECI.dyn_rho_R = self.dyn_rho_R_GECI.ctypes.data_as(POINTER(c_complex))
        GECI.prob = self.prob_GECI.ctypes.data_as(POINTER(c_double))

        #  ----------------------------- CREATING ChR2 ------------------------  #

        ChR2.nDIM = len(self.energies_ChR2)
        ChR2.energies = self.energies_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.matrix_gamma_pd = self.matrix_gamma_pd.ctypes.data_as(POINTER(c_double))
        ChR2.matrix_gamma_dep = self.matrix_gamma_dep_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.gamma_dep = self.gamma_dep_ChR2
        ChR2.frequency_A = self.frequency_A_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.freqDIM_A = len(self.frequency_A_ChR2)
        ChR2.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        ChR2.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        ChR2.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        ChR2.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))

        ChR2.rho = self.rho_ChR2.ctypes.data_as(POINTER(c_complex))
        ChR2.abs_spectra = self.abs_spectra_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.abs_dist = self.abs_dist_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.ref_spectra = self.ref_spectra_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.Raman_levels = self.Raman_levels_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.levels = self.levels_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.dyn_rho_A = self.dyn_rho_A_ChR2.ctypes.data_as(POINTER(c_complex))
        ChR2.dyn_rho_R = self.dyn_rho_R_ChR2.ctypes.data_as(POINTER(c_complex))
        ChR2.prob = self.prob_ChR2.ctypes.data_as(POINTER(c_double))

    def create_parameters_spectra(self, spectra_params, params):
        """
        Creates parameters from class parameters
        """
        spectra_params.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        spectra_params.nDIM = len(self.energies_GECI)
        spectra_params.N_exc = params.N_exc
        spectra_params.time_A = self.time_A.ctypes.data_as(POINTER(c_double))
        spectra_params.time_R = self.time_R.ctypes.data_as(POINTER(c_double))
        spectra_params.timeAMP_A = params.timeAMP_A
        spectra_params.timeAMP_R = params.timeAMP_R
        spectra_params.timeDIM_A = len(self.time_A)
        spectra_params.timeDIM_R = len(self.time_R)
        spectra_params.field_amp_A = params.field_amp_A
        spectra_params.field_amp_R = params.field_amp_R
        spectra_params.omega_R = params.omega_R
        spectra_params.omega_v = params.omega_v
        spectra_params.omega_e = params.omega_e
        spectra_params.d_alpha = params.control_guess[-1]
        spectra_params.thread_num = params.num_threads
        spectra_params.prob_guess_num = len(self.prob_GECI)
        spectra_params.spectra_lower = params.spectra_lower.ctypes.data_as(POINTER(c_double))
        spectra_params.spectra_upper = params.spectra_upper.ctypes.data_as(POINTER(c_double))
        spectra_params.max_iter = params.max_iter
        spectra_params.control_guess = params.control_guess.ctypes.data_as(POINTER(c_double))
        spectra_params.control_lower = params.control_lower.ctypes.data_as(POINTER(c_double))
        spectra_params.control_upper = params.control_upper.ctypes.data_as(POINTER(c_double))
        spectra_params.guess_num = len(params.control_guess)
        spectra_params.max_iter_control = params.max_iter_control

    def control_molA_over_molB(self, params):
        GECI = Molecule()
        ChR2 = Molecule()
        self.create_molecules(GECI, ChR2)
        params_spectra = Parameters()
        self.create_parameters_spectra(params_spectra, params)

        CalculateControl(GECI, ChR2, params_spectra)

    def control_molB_over_molA(self, params):
        GECI = Molecule()
        ChR2 = Molecule()

        self.create_molecules(GECI, ChR2)
        params_spectra = Parameters()
        self.create_parameters_spectra(params_spectra, params)

        CalculateControl(ChR2, GECI, params_spectra)


def get_experimental_spectra(mol):
    """
    Calculates interpolated linear spectra data from spectra file.
    :param mol: Spectra file of molecule mol.
    :return: Wavelength (bandwidth for specific molecule),
             Interpolated linear spectra
    """

    data = pd.read_csv(mol, sep=',')
    wavelength = data.values[:, 0]

    absorption = data.values[:, 1]

    func = interp1d(wavelength, absorption, kind='quadratic')
    wavelength_new = 1. / np.linspace(1. / wavelength.max(), 1. / wavelength.min(), 100)
    absorption_new = func(wavelength_new)
    absorption_new *= 100. / absorption_new.max()

    return wavelength_new, absorption_new


def render_ticks(axis, labelsize):
    """
    Style plots for better representation
    :param axis: axes class of plot
    """
    plt.rc('font', weight='bold')
    axis.get_xaxis().set_tick_params(
        which='both', direction='in', width=1.25, labelrotation=0, labelsize=labelsize)
    axis.get_yaxis().set_tick_params(
        which='both', direction='in', width=1.25, labelcolor='k', labelsize=labelsize)
    axis.grid(color='r', linestyle='--', linewidth=0.5, alpha=0.5, b=None, which='both', axis='both')
    # axis.get_xaxis().set_visible(False)
    # axis.get_yaxis().set_visible(False)


def dyn_plot(axes, time_R, time_A, dyn_rho_R, dyn_rho_A, mol_str):

    axes.plot(time_R, dyn_rho_R[0], 'b', label='$\\rho_{g, \\nu=0}$', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[0], 'b', linewidth=2.5)
    # axes.plot(time_R, dyn_rho_R[3], 'r', label='g4', linewidth=1.)
    # axes.plot(time_A, dyn_rho_A[3], 'r', linewidth=2.5)
    axes.plot(time_R, dyn_rho_R[4:].sum(axis=0), 'k', label='$\\rho_{e, total}$', linewidth=1.)
    axes.plot(time_A, dyn_rho_A[4:].sum(axis=0), 'k', linewidth=2.5)
    axes.set_ylabel(mol_str, fontweight='bold', fontsize='medium')


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import pickle
    from scipy.signal import savgol_filter
    cm_inv2eV_factor = 0.00012398

    # ---------------------------------------------------------------------------- #
    #                             LIST OF CONSTANTS                                #
    # ---------------------------------------------------------------------------- #
    energy_factor = 1. / 27.211385
    time_factor = .02418884 / 1000
    wavelength_freq_factor = 1239.84

    # ---------------------------------------------------------------------------- #
    #                  OBTAIN RELEVANT INFORMATION FROM SPECTRA FILES              #
    # ---------------------------------------------------------------------------- #

    #  ----------- READING WAVELENGTH AND LINEAR SPECTRA FROM FILE --------------  #
    wavelength_GECI, absorption_GECI = get_experimental_spectra('Data/GCaMP.csv')
    wavelength_ChR2, absorption_ChR2 = get_experimental_spectra("Data/ChR2_2.csv")

    absorption_GECI = savgol_filter(absorption_GECI, 5, 3)
    absorption_ChR2 = savgol_filter(absorption_ChR2, 15, 3)

    frequency_A_GECI = wavelength_freq_factor * energy_factor / wavelength_GECI
    frequency_A_ChR2 = wavelength_freq_factor * energy_factor / wavelength_ChR2

    # ---------------------------------------------------------------------------- #
    #                      GENERATE MOLECULE PARAMETERS AND MATRICES               #
    # ---------------------------------------------------------------------------- #

    #  ----------------------------- MOLECULAR CONSTANTS ------------------------  #

    N = 8  # NUMBER OF ENERGY LEVELS PER SYSTEM
    M = 11  # NUMBER OF SYSTEMS PER ENSEMBLE

    N_vib = 4  # NUMBER OF VIBRATIONAL ENERGY LEVELS IN THE GROUND STATE
    N_exc = N - N_vib  # NUMBER OF VIBRATIONAL ENERGY LEVELS IN THE EXCITED STATE

    mu_value = 2.  # VALUE OF TRANSITION DIPOLE MATRIX ELEMENTS (TIMES 2.5 DEBYE)
    gamma_pd = 2.418884e-8  # POPULATION DECAY GAMMA
    gamma_dep_GECI = 2.00 * 2.418884e-4  # DEPHASING GAMMA FOR GECI
    gamma_dep_ChR2 = 2.50 * 2.418884e-4  # DEPHASING GAMMA FOR ChR2
    gamma_vib = 0.1 * 2.418884e-5  # VIBRATIONAL DEPHASING GAMMA

    #  ------------------------ MOLECULAR MATRICES & VECTORS --------------------  #

    energies_GECI = np.empty(N)
    energies_ChR2 = np.empty(N)

    levels_GECI = np.asarray(1239.84 * energy_factor / np.linspace(400, 507, 4 * M)[::-1])  # GECI
    levels_ChR2 = np.asarray(1239.84 * energy_factor / np.linspace(370, 540, 4 * M)[::-1])  # ChR2

    rho_0 = np.zeros((N, N), dtype=np.complex)
    rho_0[0, 0] = 1 + 0j

    mu = mu_value * np.ones_like(rho_0)
    np.fill_diagonal(mu, 0j)

    matrix_gamma_pd = np.ones((N, N)) * gamma_pd
    np.fill_diagonal(matrix_gamma_pd, 0.0)
    matrix_gamma_pd = np.tril(matrix_gamma_pd).T

    matrix_gamma_dep = np.ones_like(matrix_gamma_pd) * gamma_vib
    np.fill_diagonal(matrix_gamma_dep, 0.0)

    # prob_GECI = np.asarray([0.9999450, 0.998778, 0.672025, 0.636251, 0.531511, 0.328729, 0.225944, 0.144913, 0.0689057, 0.0263029, 0.0752281])  # GECI-updated
    # prob_ChR2 = np.asarray([0.0375477, 0.150588, 0.417190, 0.685970, 0.989216, 1.000000, 0.999991, 0.802483, 0.5997950, 0.4237150, 0.3512940])  # ChR2-updated

    prob_GECI = np.asarray(
        [0.21236871, 0.21212086, 0.14272493, 0.13512723, 0.11288251, 0.06981559, 0.04798607, 0.03077668, 0.01463422,
         0.00558622, 0.01597697])  # GECI-updated
    prob_ChR2 = np.asarray(
        [0.00581433, 0.02331881, 0.0646026, 0.10622365, 0.15318182, 0.15485174, 0.15485035, 0.12426589, 0.0928793,
         0.06561301, 0.05439849])  # ChR2-updated

    spectra_lower = np.zeros(M)
    spectra_upper = np.ones(M)

    Raman_levels_GECI = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor
    Raman_levels_ChR2 = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor * 0.985

    params = ADict(

        N_exc=N_exc,
        num_threads=cpu_count(),

        energy_factor=energy_factor,
        time_factor=time_factor,
        rho_0=rho_0,

        timeDIM_R=6000,
        timeAMP_R=64277,
        timeDIM_A=3000,
        timeAMP_A=7218,

        field_amp_R=0.000235,
        field_amp_A=0.0008,

        omega_R=0.75 * energy_factor,
        omega_v=Raman_levels_GECI[3] * 0.996,
        omega_e=1239.84 * energy_factor / 545,

        spectra_lower=spectra_lower,
        spectra_upper=spectra_upper,

        max_iter=1,
        control_guess=np.asarray([0.000286828, 0.000175103, 0.015269, 0.00726709, 0.0834496, 600, 7244.77, 62620.1]),  # GECI-ChR2-----14.75
        control_lower=np.asarray([0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_GECI[3] * 0.990, 1239.84 * energy_factor / 557.5, 600, 5000, 50000]),
        control_upper=np.asarray([0.001, 0.001, 1.15 * energy_factor, Raman_levels_GECI[3] * 1.010, 1239.84 * energy_factor / 496.5, 600, 7500, 75000]),

        max_iter_control=1,
    )

    Systems = dict(
        # Constant Parameters
        matrix_gamma_pd=matrix_gamma_pd,
        matrix_gamma_dep=matrix_gamma_dep,
        mu=mu,

        # GECI molecule
        energies_GECI=energies_GECI,
        gamma_dep_GECI=gamma_dep_GECI,
        prob_GECI=prob_GECI,
        frequency_A_GECI=np.ascontiguousarray(frequency_A_GECI),
        ref_spectra_GECI=np.ascontiguousarray(absorption_GECI),
        Raman_levels_GECI=Raman_levels_GECI,
        levels_GECI=levels_GECI,

        # ChR2 molecule
        energies_ChR2=energies_ChR2,
        gamma_dep_ChR2=gamma_dep_ChR2,
        prob_ChR2=prob_ChR2,
        frequency_A_ChR2=np.ascontiguousarray(frequency_A_ChR2),
        ref_spectra_ChR2=np.ascontiguousarray(absorption_ChR2),
        Raman_levels_ChR2=Raman_levels_ChR2,
        levels_ChR2=levels_ChR2
    )

    np.set_printoptions(precision=6)
    molecule = RamanOpticalControl(params, **Systems)
    molecule.control_molA_over_molB(params)

    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 6))
    fig.canvas.set_window_title('GCaMP-ChR2')
    molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    time_axis = time_factor * (molecule.time_R.max() + np.concatenate((molecule.time_R, molecule.time_A)))
    time_R = time_factor * (molecule.time_R.max() + molecule.time_R)
    time_A = time_factor * (molecule.time_R.max() + molecule.time_A)

    axes[0, 1].plot(time_factor * (molecule.time_R.max() + molecule.time_R), 3.55e7 * molecule.field_R.real.max() * molecule.field_R.real, 'k', linewidth=1.)
    axes[0, 1].plot(time_factor * (molecule.time_R.max() + molecule.time_A), 3.55e7 * molecule.field_A.real.max() * molecule.field_A.real, 'darkblue', linewidth=1.)

    dyn_plot(axes[1, 1], time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, '')
    dyn_plot(axes[2, 1], time_R, time_A, molecule.dyn_rho_R_ChR2.real, molecule.dyn_rho_A_ChR2.real, '')

    axes[1, 1].plot(time_R, molecule.dyn_rho_R_GECI.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    axes[1, 1].plot(time_A, molecule.dyn_rho_A_GECI.real[3], 'r', linewidth=2.5)
    axes[2, 1].plot(time_R, molecule.dyn_rho_R_ChR2.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    axes[2, 1].plot(time_A, molecule.dyn_rho_A_ChR2.real[3], 'r', linewidth=2.5)

    axes[2, 0].set_xlabel('Time (in ps)', fontweight='bold', fontsize='medium')
    axes[2, 1].set_xlabel('Time (in ps)', fontweight='bold', fontsize='medium')
    axes[0, 0].set_ylabel('Electric field \n (in $GW/cm^2$)', fontweight='bold')

    axes[1, 1].legend(loc=6, prop={'weight': 'normal', 'size': 'small'})
    axes[2, 1].legend(loc=6, prop={'weight': 'normal', 'size': 'small'})

    axes[0, 1].set_xlim(0, 2*time_factor*(params.control_guess[-1] + params.control_guess[-2]))
    axes[1, 1].set_xlim(0, 2*time_factor*(params.control_guess[-1] + params.control_guess[-2]))
    axes[2, 1].set_xlim(0, 2*time_factor*(params.control_guess[-1] + params.control_guess[-2]))
    
    render_ticks(axes[0, 1], 'large')
    render_ticks(axes[1, 1], 'large')
    render_ticks(axes[2, 1], 'large')

    for i in range(3):
        axes[i][0].set_yticklabels([])
        axes[i][1].set_yticklabels([])

    print(molecule.rho_GECI.diagonal()[4:].sum().real)
    print(molecule.rho_ChR2.diagonal()[4:].sum().real)
    print(molecule.rho_GECI.diagonal()[4:].sum().real/molecule.rho_ChR2.diagonal()[4:].sum().real)

    del molecule

    params.control_guess = np.asarray([0, 0.000498823, 0.0374115, 0.00727501, 0.0913695, 600, 7467.52, 0.1])
    params.control_lower = np.asarray([0, 0.0001, 0.35 * energy_factor, Raman_levels_GECI[3] * 0.990, 1239.84 * energy_factor / 557.5, 600, 5000, 0.1])
    params.control_upper = np.asarray([0, 0.0005, 1.15 * energy_factor, Raman_levels_GECI[3] * 1.010, 1239.84 * energy_factor / 406.5, 600, 7500, 0.1])

    params.max_iter_control = 1

    molecule = RamanOpticalControl(params, **Systems)
    molecule.control_molA_over_molB(params)

    molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    time_axis = time_factor * (molecule.time_R.max() + np.concatenate((molecule.time_R, molecule.time_A)))
    time_A = time_factor * (molecule.time_R.max() + molecule.time_A)

    axes[0, 0].plot(time_factor * (molecule.time_R.max() + molecule.time_A), 3.55e7 * molecule.field_A.real.max() * molecule.field_A.real,
                    'darkblue', linewidth=1.)

    dyn_plot(axes[1, 0], time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, '')
    dyn_plot(axes[2, 0], time_R, time_A, molecule.dyn_rho_R_ChR2.real, molecule.dyn_rho_A_ChR2.real, '')

    axes[2, 0].set_xlabel('Time (in ps)', fontweight='bold', fontsize='medium')

    axes[1, 0].legend(loc=6, prop={'weight': 'normal', 'size': 'small'})
    axes[2, 0].legend(loc=6, prop={'weight': 'normal', 'size': 'small'})

    axes[0, 0].set_xlim(0, 2 * time_factor * (params.control_guess[-1] + params.control_guess[-2]))
    axes[1, 0].set_xlim(0, 2 * time_factor * (params.control_guess[-1] + params.control_guess[-2]))
    axes[2, 0].set_xlim(0, 2 * time_factor * (params.control_guess[-1] + params.control_guess[-2]))
    render_ticks(axes[0, 0], 'large')
    render_ticks(axes[1, 0], 'large')
    render_ticks(axes[2, 0], 'large')

    print(molecule.rho_GECI.diagonal()[4:].sum().real)
    print(molecule.rho_ChR2.diagonal()[4:].sum().real)
    print(molecule.rho_ChR2.diagonal()[4:].sum().real / molecule.rho_GECI.diagonal()[4:].sum().real)

    for i in range(2):
        axes[i][0].set_xticklabels([])
        axes[i][1].set_xticklabels([])
        axes[i][2].set_xticklabels([])

    axes[1, 0].set_ylabel("Population \n of GCaMP", fontweight='bold', fontsize='medium')
    axes[2, 0].set_ylabel("Population \n of ChR2", fontweight='bold', fontsize='medium')

    axes[0, 0].yaxis.set_ticks_position('right')
    axes[1, 0].yaxis.set_ticks_position('right')
    axes[2, 0].yaxis.set_ticks_position('right')

    del molecule

    params.control_guess = np.asarray(
        [0.000266538, 0.000228214, 0.0176864, 0.00715798, 0.0853147, 600, 6350.12, 68957.5,])
    params.control_lower = np.asarray(
        [0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_ChR2[3] * 0.990, 1239.84 * energy_factor / 550.5, 600, 5000,
         50000])
    params.control_upper = np.asarray(
        [0.001, 0.001, 1.15 * energy_factor, Raman_levels_ChR2[3] * 1.010, 1239.84 * energy_factor / 476.5, 600, 7500,
         75000])

    params.max_iter_control = 1

    molecule = RamanOpticalControl(params, **Systems)
    molecule.control_molB_over_molA(params)

    molecule.time_A += molecule.time_R.max() + molecule.time_A.max()
    time_axis = time_factor * (molecule.time_R.max() + np.concatenate((molecule.time_R, molecule.time_A)))
    time_R = time_factor * (molecule.time_R.max() + molecule.time_R)
    time_A = time_factor * (molecule.time_R.max() + molecule.time_A)

    axes[0, 2].plot(time_factor * (molecule.time_R.max() + molecule.time_R),
                    3.55e7 * molecule.field_R.real.max() * molecule.field_R.real, 'k', linewidth=1.)
    axes[0, 2].plot(time_factor * (molecule.time_R.max() + molecule.time_A),
                    3.55e7 * molecule.field_A.real.max() * molecule.field_A.real,
                    'darkblue', linewidth=1.)

    dyn_plot(axes[1, 2], time_R, time_A, molecule.dyn_rho_R_GECI.real, molecule.dyn_rho_A_GECI.real, '')
    dyn_plot(axes[2, 2], time_R, time_A, molecule.dyn_rho_R_ChR2.real, molecule.dyn_rho_A_ChR2.real, '')

    axes[1, 2].plot(time_R, molecule.dyn_rho_R_GECI.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    axes[1, 2].plot(time_A, molecule.dyn_rho_A_GECI.real[3], 'r', linewidth=2.5)
    axes[2, 2].plot(time_R, molecule.dyn_rho_R_ChR2.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    axes[2, 2].plot(time_A, molecule.dyn_rho_A_ChR2.real[3], 'r', linewidth=2.5)

    axes[2, 2].set_xlabel('Time (in ps)', fontweight='bold', fontsize='medium')
    # axes[0, 2].ticklabel_format(style='sci', scilimits=(0, 3))

    axes[1, 2].legend(loc=6, prop={'weight': 'normal', 'size': 'small'})
    axes[2, 2].legend(loc=6, prop={'weight': 'normal', 'size': 'small'})

    axes[0, 2].set_xlim(0, 2 * time_factor * (params.control_guess[-1] + params.control_guess[-2]))
    axes[1, 2].set_xlim(0, 2 * time_factor * (params.control_guess[-1] + params.control_guess[-2]))
    axes[2, 2].set_xlim(0, 2 * time_factor * (params.control_guess[-1] + params.control_guess[-2]))
    render_ticks(axes[0, 2], 'large')
    render_ticks(axes[1, 2], 'large')
    render_ticks(axes[2, 2], 'large')

    print(molecule.rho_GECI.diagonal()[4:].sum().real)
    print(molecule.rho_ChR2.diagonal()[4:].sum().real)
    print(molecule.rho_ChR2.diagonal()[4:].sum().real / molecule.rho_GECI.diagonal()[4:].sum().real)

    for i in range(3):
        axes[0][i].set_ylim(-14, 14)

    for i in range(2):
        axes[i][0].set_xticklabels([])
        axes[i][1].set_xticklabels([])
        axes[i][2].set_xticklabels([])

    axes[0, 2].yaxis.set_ticks_position('right')
    axes[1, 2].yaxis.set_ticks_position('right')
    axes[2, 2].yaxis.set_ticks_position('right')

    fig.subplots_adjust(bottom=0.15, top=0.96, left=0.05, hspace=0.1, wspace=0.1)
    fig.text(0.265, 0.05, '(A)', horizontalalignment='center', verticalalignment='center', weight='bold').set_zorder(20)
    fig.text(0.530, 0.05, '(B)', horizontalalignment='center', verticalalignment='center', weight='bold').set_zorder(20)
    fig.text(0.785, 0.05, '(C)', horizontalalignment='center', verticalalignment='center', weight='bold').set_zorder(20)
    # plt.savefig('FinalPaperPlots/GECI_ChR2_dynamics.eps', format="eps")
    # plt.savefig('FinalPaperPlots/GECI_ChR2_dynamics.png', format="png")
    # plt.show()

    # import time
    # start = time.time()
    # params.max_iter_control = 45
    # N_points = 15
    # Z = np.empty((N_points, N_points))
    # X = np.linspace(0.985, 1.0, N_points, endpoint=True)
    # Y = 1 / np.linspace(.1, 1, N_points, endpoint=True)
    # for x, i in enumerate(X):
    #     for y, j in enumerate(Y):
    #         Raman_levels_GECI = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor
    #         Raman_levels_ChR2 = np.asarray([0, 1000, 1300, 1600]) * energy_factor * cm_inv2eV_factor * i
    #
    #         gamma_vib = 2.418884e-5 / j
    #         matrix_gamma_dep = np.ones_like(matrix_gamma_pd) * gamma_vib
    #         np.fill_diagonal(matrix_gamma_dep, 0.0)
    #
    #         params.control_guess = np.asarray([0.000295219, 0.000219229, 0.0140962, Raman_levels_GECI[3], 0.0837432, 600, 7218.48, 64277.3])  # GECI-ChR2-----14.75
    #         params.control_lower = np.asarray([0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_GECI[3] * 0.990, 1239.84 * energy_factor / 557.5, 600, 5000, 50000])
    #         params.control_upper = np.asarray([0.001, 0.001, 1.15 * energy_factor, Raman_levels_GECI[3] * 1.010, 1239.84 * energy_factor / 496.5, 600, 7500, 75000])
    #
    #         Systems['matrix_gamma_dep'] = matrix_gamma_dep
    #         Systems['Raman_levels_GECI'] = Raman_levels_GECI
    #         Systems['Raman_levels_ChR2'] = Raman_levels_ChR2
    #
    #         molecule = RamanOpticalControl(params, **Systems)
    #         molecule.control_molA_over_molB(params)
    #
    #         Z[x][y] = molecule.rho_GECI.diagonal()[4:].sum().real / molecule.rho_ChR2.diagonal()[4:].sum().real
    #         print('\n', x, y, i, j, Z[x][y], '\n')
    #         del molecule
    #
    # print(time.time() - start)
    # pickle.dump(Z, open("discrimination_GECI_GECI_vib.p", "wb"))
    # Z_read = pickle.load(open("discrimination_GECI_GECI_vib.p", "rb"))
    #
    # X *= 1600
    #
    # fig, axes = plt.subplots(nrows=1, ncols=1)
    # axes.imshow(Z, interpolation='nearest', cmap=cm.hot)
    # plt.imshow(X, 1/Y, Z, extent=[X.min(), X.max(), Y.max(), Y.min()])

    plt.show()
