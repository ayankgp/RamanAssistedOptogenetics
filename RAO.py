#!/usr/bin/env python

"""
RAO.py:

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

        self.delay = params.delay

        if params.delay > params.timeAMP_A:
            self.timeAMP = params.timeAMP_R
        else:
            self.timeAMP = params.timeAMP_R + params.timeAMP_A - self.delay

        self.time = np.linspace(0, self.timeAMP, params.timeDIM)

        self.field_R = np.zeros(params.timeDIM, dtype=np.complex)
        self.field_A = np.zeros(params.timeDIM, dtype=np.complex)
        self.field = np.zeros(params.timeDIM, dtype=np.complex)

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

        self.dyn_rho_GECI = np.ascontiguousarray(np.zeros((N, params.timeDIM)), dtype=np.complex)
        self.dyn_rho_ChR2 = np.ascontiguousarray(np.zeros((N, params.timeDIM)), dtype=np.complex)

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
        GECI.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))
        GECI.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        GECI.field = self.field.ctypes.data_as(POINTER(c_complex))

        GECI.rho = self.rho_GECI.ctypes.data_as(POINTER(c_complex))
        GECI.abs_spectra = self.abs_spectra_GECI.ctypes.data_as(POINTER(c_double))
        GECI.abs_dist = self.abs_dist_GECI.ctypes.data_as(POINTER(c_double))
        GECI.ref_spectra = self.ref_spectra_GECI.ctypes.data_as(POINTER(c_double))
        GECI.Raman_levels = self.Raman_levels_GECI.ctypes.data_as(POINTER(c_double))
        GECI.levels = self.levels_GECI.ctypes.data_as(POINTER(c_double))
        GECI.dyn_rho = self.dyn_rho_GECI.ctypes.data_as(POINTER(c_complex))
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
        ChR2.field_R = self.field_R.ctypes.data_as(POINTER(c_complex))
        ChR2.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        ChR2.field = self.field.ctypes.data_as(POINTER(c_complex))

        ChR2.rho = self.rho_ChR2.ctypes.data_as(POINTER(c_complex))
        ChR2.abs_spectra = self.abs_spectra_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.abs_dist = self.abs_dist_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.ref_spectra = self.ref_spectra_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.Raman_levels = self.Raman_levels_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.levels = self.levels_ChR2.ctypes.data_as(POINTER(c_double))
        ChR2.dyn_rho = self.dyn_rho_ChR2.ctypes.data_as(POINTER(c_complex))
        ChR2.prob = self.prob_ChR2.ctypes.data_as(POINTER(c_double))

    def create_parameters_spectra(self, spectra_params, params):
        """
        Creates parameters from class parameters
        """
        spectra_params.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        spectra_params.nDIM = len(self.energies_GECI)
        spectra_params.N_exc = params.N_exc
        spectra_params.time = self.time.ctypes.data_as(POINTER(c_double))
        spectra_params.timeAMP_A = params.timeAMP_A
        spectra_params.timeAMP_R = params.timeAMP_R
        spectra_params.timeAMP = self.timeAMP
        spectra_params.timeDIM = len(self.time)
        spectra_params.delay = self.delay
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


def dyn_plot(axes, time, dyn_rho, mol_str):
    axes.plot(time, dyn_rho[0], 'b', label='$\\rho_{g, \\nu=0}$', linewidth=1.)
    axes.plot(time, dyn_rho[4:].sum(axis=0), 'k', label='$\\rho_{e, total}$', linewidth=1.)
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
    wavelength_ChR2, absorption_ChR2 = get_experimental_spectra("Data/ChR2.csv")

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

        timeAMP_R=64277 * 2,
        timeAMP_A=7218 * 2,
        delay=-50000,
        timeDIM=10000,

        field_amp_R=0.000235,
        field_amp_A=0.0008,

        omega_R=0.75 * energy_factor,
        omega_v=Raman_levels_GECI[3] * 0.996,
        omega_e=1239.84 * energy_factor / 545,

        spectra_lower=spectra_lower,
        spectra_upper=spectra_upper,

        max_iter=1,
        control_guess=np.asarray([0.000286828, 0.000175103, 0.015269, 0.00726709, 0.0834496, 600, 2*7244.77, 2*62620.1]),
        # GECI-ChR2-----14.75
        control_lower=np.asarray(
            [0.0001, 0.0001, 0.35 * energy_factor, Raman_levels_GECI[3] * 0.990, 1239.84 * energy_factor / 557.5, 600,
             10000, 100000]),
        control_upper=np.asarray(
            [0.001, 0.001, 1.15 * energy_factor, Raman_levels_GECI[3] * 1.010, 1239.84 * energy_factor / 496.5, 600,
             15000, 150000]),

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

    axes[0, 1].plot(time_factor * molecule.time, 3.55e7 * molecule.field.max() * molecule.field.real, 'k', linewidth=1.)
    axes[0, 1].plot(time_factor * molecule.time, 3.55e7 * molecule.field.max() * molecule.field_R.real, 'b', linewidth=1.)
    axes[0, 1].plot(time_factor * molecule.time, 3.55e7 * molecule.field.max() * molecule.field_A.real, 'r', linewidth=1.)

    dyn_plot(axes[1, 1], time_factor * molecule.time, molecule.dyn_rho_GECI.real, '')
    dyn_plot(axes[2, 1], time_factor * molecule.time, molecule.dyn_rho_ChR2.real, '')
    axes[1, 1].plot(time_factor * molecule.time, molecule.dyn_rho_GECI.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)
    axes[2, 1].plot(time_factor * molecule.time, molecule.dyn_rho_ChR2.real[3], 'r', label='$\\rho_{g, \\nu=R}$', linewidth=1.)

    axes[2, 0].set_xlabel('Time (in ps)', fontweight='bold', fontsize='medium')
    axes[2, 1].set_xlabel('Time (in ps)', fontweight='bold', fontsize='medium')
    axes[0, 0].set_ylabel('Electric field \n (in $GW/cm^2$)', fontweight='bold')

    axes[1, 1].legend(loc=6, prop={'weight': 'normal', 'size': 'small'})
    axes[2, 1].legend(loc=6, prop={'weight': 'normal', 'size': 'small'})

    axes[0, 1].set_xlim(0, time_factor * molecule.time.max())
    axes[1, 1].set_xlim(0, time_factor * molecule.time.max())
    axes[2, 1].set_xlim(0, time_factor * molecule.time.max())

    render_ticks(axes[0, 1], 'large')
    render_ticks(axes[1, 1], 'large')
    render_ticks(axes[2, 1], 'large')

    N_delay = 80
    delay_axis = np.linspace(120000, -10000, N_delay)
    delay_dep = np.zeros_like(delay_axis)

    for i in range(N_delay):
        del molecule
        params.delay = delay_axis[i]

        molecule = RamanOpticalControl(params, **Systems)
        molecule.control_molA_over_molB(params)

        delay_dep[i] = molecule.rho_GECI.diagonal().real[4:].sum() / molecule.rho_ChR2.diagonal().real[4:].sum()
        print(i, delay_axis[i], delay_dep[i])

    fig_delay, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(-time_factor * delay_axis, delay_dep, 'r', linewidth=2.)
    render_ticks(ax, 'x-large')
    ax.set_xlabel('Delay between Raman \n and electronic fields (in ps)', fontweight='bold', fontsize='medium')
    ax.set_ylabel('Ratio of excited state populations \n of GECI to ChR2 ($\\rho_{exc}^{GECI} / \\rho_{exc}^{ChR2}$)',
                  fontweight='bold', fontsize='medium')
    plt.show()
