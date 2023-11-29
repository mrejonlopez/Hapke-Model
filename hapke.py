import math
import random
import numpy as np
from matplotlib.collections import PatchCollection
from scipy import optimize
from scipy.interpolate import interp1d
import os
from pyvims import VIMS
import shutil
from matplotlib.patches import Polygon
from pyvims.misc import MAPS
from scipy.optimize import curve_fit
from scipy.integrate import simps

random.seed(11770)

density_cr = 0.931
density_am = 0.7


# DECIDED TO TAKE THE SAME DENSITY AS THE DIFFERENCE IS VERY SMALL

###############################################################################
###############################################################################
###############################################################################
####################### MAIN HAPKE FUNCTIONS ##################################
###############################################################################
###############################################################################
###############################################################################

def hapke_model_mixed(parameters, wav, angles, n_c, k_c, n_am, k_am, body='0'):
    """
    :param body: icy moon to be considered
    :param parameters: to be optimized, in order filling factor, grain size and surface roughness
    :param wav: wavelength array
    :param angles: in order incidence, emergence and phase
    :param n_c & n_am: n array of crystalline and amorphous ice
    :param k_c & k_am: k array of crystalline and amorphous ice
    :return:
    """

    mass_fraction = 0
    b = 0.3
    phi = 0.4
    #theta_bar = np.deg2rad(25)
    theta_bar, D = parameters

    eme, inc, phase = angles
    # Calculate the modeled spectrum using Hapke model equations
    psi = phasetoazimuth(phase, eme, inc)
    w_c = singlescatteringalbedo(n_c, k_c, wav, D)
    w_am = singlescatteringalbedo(n_am, k_am, wav, D)

    c = hockey_stick(b)
    p = phase_function(b, c, phase)

    R0_c = fresnel_coefficient(n_c, k_c)
    R0_am = fresnel_coefficient(n_am, k_am)

    B_S0 = shoe_amp_mix(mass_fraction, wav, wav, p, p, R0_c, R0_am)

    K = porosityparameter(phi)

    B_SH = shoe(B_S0, phi, phase)

    [mu_0e, mu_e, S] = shadowingfunction(inc, eme, psi, theta_bar)

    w = singlescatteringalbedomixed(mass_fraction, w_c, w_am)

    r = np.empty(len(wav))

    for i in range(len(wav)):
        r[i] = (K * w[i] / (4 * np.pi) * mu_0e / (mu_0e + mu_e) * (
                p * B_SH[i] + H(w[i], mu_0e) * H(w[i], mu_e) - 1) * S)

    IF = r * np.pi

    output = {'r': r, 'IF': IF, 'w': w, 'p': p}

    return output

def hapke_model_mixed_b(parameters, wav, angles, n_c, k_c, n_am, k_am, b):
    """
    :param body: icy moon to be considered
    :param parameters: to be optimized, in order filling factor, grain size and surface roughness
    :param wav: wavelength array
    :param angles: in order incidence, emergence and phase
    :param n_c & n_am: n array of crystalline and amorphous ice
    :param k_c & k_am: k array of crystalline and amorphous ice
    :return:
    """

    mass_fraction = 0

    phi = 0.4
    #theta_bar = np.deg2rad(25)
    theta_bar, D = parameters

    eme, inc, phase = angles
    # Calculate the modeled spectrum using Hapke model equations
    psi = phasetoazimuth(phase, eme, inc)
    w_c = singlescatteringalbedo(n_c, k_c, wav, D)
    w_am = singlescatteringalbedo(n_am, k_am, wav, D)

    c = hockey_stick(b)
    p = phase_function(b, c, phase)

    R0_c = fresnel_coefficient(n_c, k_c)
    R0_am = fresnel_coefficient(n_am, k_am)

    B_S0 = shoe_amp_mix(mass_fraction, wav, wav, p, p, R0_c, R0_am)

    K = porosityparameter(phi)

    B_SH = shoe(B_S0, phi, phase)

    [mu_0e, mu_e, S] = shadowingfunction(inc, eme, psi, theta_bar)

    w = singlescatteringalbedomixed(mass_fraction, w_c, w_am)

    r = np.empty(len(wav))

    for i in range(len(wav)):
        r[i] = (K * w[i] / (4 * np.pi) * mu_0e / (mu_0e + mu_e) * (
                p * B_SH[i] + H(w[i], mu_0e) * H(w[i], mu_e) - 1) * S)

    IF = r * np.pi

    output = {'r': r, 'IF': IF, 'w': w, 'p': p}

    return output


def hapke_model_mixed_phi(parameters, wav, angles, n_c, k_c, n_am, k_am, phi):
    """
    :param body: icy moon to be considered
    :param parameters: to be optimized, in order filling factor, grain size and surface roughness
    :param wav: wavelength array
    :param angles: in order incidence, emergence and phase
    :param n_c & n_am: n array of crystalline and amorphous ice
    :param k_c & k_am: k array of crystalline and amorphous ice
    :return:
    """

    mass_fraction = 0
    b = 0.3

    # theta_bar = np.deg2rad(25)
    theta_bar, D = parameters

    eme, inc, phase = angles
    # Calculate the modeled spectrum using Hapke model equations
    psi = phasetoazimuth(phase, eme, inc)
    w_c = singlescatteringalbedo(n_c, k_c, wav, D)
    w_am = singlescatteringalbedo(n_am, k_am, wav, D)

    c = hockey_stick(b)
    p = phase_function(b, c, phase)

    R0_c = fresnel_coefficient(n_c, k_c)
    R0_am = fresnel_coefficient(n_am, k_am)

    B_S0 = shoe_amp_mix(mass_fraction, wav, wav, p, p, R0_c, R0_am)

    K = porosityparameter(phi)

    B_SH = shoe(B_S0, phi, phase)

    [mu_0e, mu_e, S] = shadowingfunction(inc, eme, psi, theta_bar)

    w = singlescatteringalbedomixed(mass_fraction, w_c, w_am)

    r = np.empty(len(wav))

    for i in range(len(wav)):
        r[i] = (K * w[i] / (4 * np.pi) * mu_0e / (mu_0e + mu_e) * (
                p * B_SH[i] + H(w[i], mu_0e) * H(w[i], mu_e) - 1) * S)

    IF = r * np.pi

    output = {'r': r, 'IF': IF, 'w': w, 'p': p}

    return output

def loc_to_m(loc2,loc2_err):
    slope = -0.01735
    slope_error = 0.00018
    intercept = 2.0294
    intercept_error = 0.0001

    x = (loc2 - intercept) / slope

    # Calculate the total uncertainty in x using error propagation
    x_error = np.sqrt((loc2_err / slope) ** 2 + ((intercept - loc2) / slope ** 2) ** 2 * slope_error ** 2 + (
                intercept_error / slope) ** 2)

    return [x,x_error]

def hapke_model_mixed_mass_fraction(parameters, wav, angles, n_c, k_c, n_am, k_am, aux_param):
    """
    :param parameters: to be optimized, in order filling factor, grain size and surface roughness
    :param wav: wavelength array
    :param angles: in order incidence, emergence and phase
    :param n_c & n_am: n array of crystalline and amorphous ice
    :param k_c & k_am: k array of crystalline and amorphous ice
    :return:
    """

    # Unpack the parameters
    # Assuming you have parameters p1, p2, p3, etc.

    b = 0.3
    phi = 0.4
    theta_bar, D = aux_param
    mass_fraction = parameters
    eme, inc, phase = angles
    # Calculate the modeled spectrum using Hapke model equations
    psi = phasetoazimuth(phase, eme, inc)
    w_c = singlescatteringalbedo(n_c, k_c, wav, D)
    w_am = singlescatteringalbedo(n_am, k_am, wav, D)

    c = hockey_stick(b)
    p = phase_function(b, c, phase)

    R0_c = fresnel_coefficient(n_c, k_c)
    R0_am = fresnel_coefficient(n_am, k_am)

    B_S0 = shoe_amp_mix(mass_fraction, wav, wav, p, p, R0_c, R0_am)

    K = porosityparameter(phi)

    B_SH = shoe(B_S0, phi, phase)

    [mu_0e, mu_e, S] = shadowingfunction(inc, eme, psi, theta_bar)

    w = singlescatteringalbedomixed(mass_fraction, w_c, w_am)

    r = np.empty(len(wav))

    for i in range(len(wav)):
        r[i] = (K * w[i] / (4 * np.pi) * mu_0e / (mu_0e + mu_e) * (
                p * B_SH[i] + H(w[i], mu_0e) * H(w[i], mu_e) - 1) * S)

    IF = r * np.pi

    output = {'r': r, 'IF': IF, 'w': w, 'p': p}

    return output

def hapke_model(parameters, wav, angles, n_range, k_range):
    """
    :param parameters: to be optimized, in order filling factor, grain size and surface roughness
    :param wav: wavelength array
    :param angles: in order incidence, emergence and phase
    :param n_range: n array
    :param k_range: k array
    :return:
    """

    # Unpack the parameters
    # Assuming you have parameters p1, p2, p3, etc.
    D, phi = parameters
    eme, inc, phase = angles

    # Calculate the modeled spectrum using Hapke model equations

    theta_bar = np.deg2rad(20.0)
    psi = phasetoazimuth(phase, eme, inc)

    b = 0.2
    c = hockey_stick(b)
    p = phase_function(b, c, phase)

    w = singlescatteringalbedo(n_range, k_range, wav, D)
    S_0 = fresnel_coefficient(n_range, k_range)
    K = porosityparameter(phi)
    B_S0 = shoe_amp(w, p, S_0)
    B_SH = shoe(B_S0, phi, phase)

    [mu_0e, mu_e, S] = shadowingfunction(inc, eme, psi, theta_bar)

    r = []
    for i in range(len(wav)):
        r.append(K * w[i] / (4 * np.pi) * mu_0e / (mu_0e + mu_e) * (
                p * B_SH[i] + H(w[i], mu_0e) * H(w[i], mu_e) - 1) * S)

    IF = np.array(r) * np.pi

    output = {'r': r, 'IF': IF, 'w': w, 'p': p}

    return output


###############################################################################
###############################################################################
###############################################################################
########################## HAPKE FUNCTIONS ####################################
###############################################################################
###############################################################################
###############################################################################


def singlescatteringalbedo(n, k, wavelength, D, internalScattering=False):
    """
    Generalized equivalent-slab model (Hapke 2012; Section 6.5)
    n & k: optical constants
    wavelength: wavelength in microns
    D: particle mean size
    internalScattering: boolean

    """

    if not internalScattering:
        s = 0
    elif internalScattering:
        s = 10000  # cm^(-1)
        s = 1 / ((1 / s) * 10 ** (-2))  # m^(-1)

    R_0 = fresnel_coefficient(n, k)

    Se = np.empty((len(n)))
    Si = np.empty((len(n)))
    alpha = np.empty((len(n)))
    Dmodif = np.empty((len(n)))
    r_i = np.empty((len(n)))
    Theta = np.empty((len(n)))
    w = np.empty((len(n)))

    for i in range(len(n)):
        Se[i] = 0.0587 + 0.8543 * R_0[i] + 0.0870 * R_0[i] ** 2
        Si[i] = 1 - (0.9413 - 0.8543 * R_0[i] + 0.0870 * R_0[i] ** 2) / n[i]

        # absorption coefficient multiplied by a factor pi
        alpha[i] = (4 * np.pi * k[i]) / (wavelength[i] * 10 ** (-6))

        Dmodif[i] = np.real((2 / 3) * (n[i] ** 2 - (1 / n[i]) * (n[i] ** 2 - 1 + 0j) ** (3 / 2)) * D)
        r_i[i] = (1 - np.sqrt(alpha[i] / (alpha[i] + s))) / (1 + np.sqrt(alpha[i] / (alpha[i] + s)))
        Theta[i] = (r_i[i] + np.exp(-np.sqrt(alpha[i] * (alpha[i] + s)) * Dmodif[i])) / (
                1 + r_i[i] * np.exp(-np.sqrt(alpha[i] * (alpha[i] + s)) * Dmodif[i]))

        # calculation of the single scattering albedo
        w[i] = Se[i] + (1 - Se[i]) * ((1 - Si[i]) / (1 - Si[i] * Theta[i])) * Theta[i]

    return w


def hockey_stick(b):
    """
    Hockey-stick function
    b: phase function asymmetry parameter
    """

    c = 3.29 * np.exp(-17.4 * b ** 2) - 0.908
    return c


def phase_function(b, c, g):
    """
    With Hockey-stick function implemented
    b: phase function asymmetry parameter
    g: phase angle in rad

    """

    P = ((1 - c) / 2) * (1 - b ** 2) / ((1 + 2 * b * np.cos(g) + b ** 2) ** (3 / 2)) + ((1 + c) / 2) * (1 - b ** 2) / (
            (1 - 2 * b * np.cos(g) + b ** 2) ** (3 / 2))

    return P


def cboe(B_C0, meanfreepath, wavelength, g, K):
    """
    B_C0: amplitude coefficient of CBOE
    g: phase angle in rad
    wavelength: wavelength in microns
    MeanFreePath:
    K: boolean

    """

    h_C = np.empty(len(wavelength))
    B_C = np.empty(len(wavelength))

    for i in range(len(wavelength)):
        # Angular width of CBOE
        h_C[i] = (wavelength[i] * 10 ** (-6)) / (4 * np.pi * meanfreepath)
        if g == 0:
            g = 10 ** (-15)
        B_C[i] = (1 + (1 - np.exp(-1.42 * K * np.tan(g / 2) / h_C[i])) / (np.tan(g / 2) / h_C[i])) / (
                ((1 + np.tan(g / 2) / h_C[i]) ** 2) * (1 + 1.42 * K))

    return np.array(1 + B_C0 * B_C)


def azimuthtophase(psi, e, i):
    return np.arccos(np.cos(e) * np.cos(i) + np.sin(e) * np.sin(i) * np.cos(psi))


def phasetoazimuth(g, e, i):
    return np.arccos((np.cos(g) - np.cos(e) * np.cos(i)) / (np.sin(e) * np.sin(i)))


def shadowingfunction(i, e, psi, thetabar):
    """
    Shadowing function S accounting for the roughness of the surface

    :param i: angle of incidence
    :param e: angle of emergence
    :param psi: azimuth angle (angle between incidence and emergence planes)
    :param thetabar: surface roughness angle
    :return: vector containing  coefficient mu_0e (= cos(i_e))
                                coefficient mu_e (= cos(e_e))
                                shadowing coefficient S(i, e, g)
    i_e, e_e : effective angles of incidence and emergence
    """

    # Auxiliary functions

    xi = 1 / np.sqrt(1 + np.pi * np.tan(thetabar) ** 2)

    def E1(x):
        if thetabar == 0 or x == 0:
            return 0
        else:
            return np.exp(-2 / (np.pi * np.tan(thetabar) * np.tan(x)))

    def E2(x):

        if thetabar == 0 or x == 0:
            return 0
        else:
            return np.exp(-2 / (np.pi * np.tan(thetabar) ** 2 * np.tan(x) ** 2))

    def fpsi(psi):
        return np.exp(-2 * np.tan(psi / 2))

    def eta(y):
        return xi * (np.cos(y) + np.sin(y) * np.tan(thetabar) * E2(y) / (2 - E1(y)))

    mu_0 = np.cos(i)
    mu = np.cos(e)
    if e >= i:
        mu_0e = xi * (np.cos(i) + np.sin(i) * np.tan(thetabar) * (
                np.cos(psi) * E2(e) + (np.sin(psi / 2)) ** 2 * E2(i)) / (2 - E1(e) - (psi / np.pi) * E1(i)))
        mu_e = xi * (np.cos(e) + np.sin(e) * np.tan(thetabar) * (E2(e) + (np.sin(psi / 2)) ** 2 * E2(i)) / (
                2 - E1(e) - (psi / np.pi) * E1(i)))
        S = (mu_e / eta(e)) * (mu_0 / eta(i)) * (xi / (1 - fpsi(psi) + fpsi(psi) * xi * (mu_0 / eta(i))))
    else:
        mu_0e = xi * (np.cos(i) + np.sin(i) * np.tan(thetabar) * (E2(i) + (np.sin(psi / 2)) ** 2 * E2(e)) / (
                2 - E1(i) - (psi / np.pi) * E1(e)))
        mu_e = xi * (np.cos(e) + np.sin(e) * np.tan(thetabar) * (
                np.cos(psi) * E2(i) + (np.sin(psi / 2)) ** 2 * E2(e)) / (2 - E1(i) - (psi / np.pi) * E1(e)))
        S = (mu_e / eta(e)) * (mu_0 / eta(i)) * (xi / (1 - fpsi(psi) + fpsi(psi) * xi * (mu / eta(e))))

    return mu_0e, mu_e, S


def H(w, x):
    """
    :param w: single scattering albedo
    :param x: either mu_0e or mu_e
    :return: Multiple scattering function

    Implementation from Hapke 2012 eq. 8.56 (section 8.7.3.3)
    """
    gamma = np.sqrt(1 - w)

    # Diffuse reflectance
    r_0 = (1 - gamma) / (1 + gamma)

    # two different formulations for the function H (approximation or
    # precise estimation)
    highPrecision = 1

    if highPrecision == 0:
        multipleScattering = (1 + 2 * x) / (1 + 2 * gamma * x)
    else:
        # OLD VERSION FROM HAPKE 1993
        # multipleScatteringFunction = (1 - (1 - gamma) * x * (r_0 + (1 - (1 / 2) * r_0 - r_0 * x) * np.log((1 + x) / x))) ** (-1)

        # UPDATED VERSION FROM HAPKE 2012
        multipleScattering = (1 - w * x * (r_0 + ((1 - 2 * r_0 * x) / 2) * np.log((1 + x) / x))) ** (-1)

    return multipleScattering


def porosityparameter(phi):
    """

    :param phi: filling factor (1-porosity)
    :return: porosity parameter
    """

    if phi == 0:
        K = 1
    else:
        K = (-np.log(1 - 1.209 * phi ** (2 / 3))) / (1.209 * phi ** (2 / 3))
    return K  # RANGES BETWEEN 1 AND 8.43


def shoe(B_S0, phi, g):
    h_s = 3 * porosityparameter(phi) * phi / 8
    B_S = (1 + np.tan(g / 2) / h_s) ** (-1)

    return np.array(1 + B_S0 * B_S)


def fresnel_coefficient(n, k):
    R0 = np.empty(len(n))

    for i in range(len(n)):
        R0[i] = ((n[i] - 1) ** 2 + k[i] ** 2) / ((n[i] + 1) ** 2 + k[i] ** 2)

    return R0


###############################################################################
###############################################################################
###############################################################################
###################### AUXILIARY FUNCTIONS ####################################
###############################################################################
###############################################################################
###############################################################################

def crystallinity_coecient(IF, wav):
    index_1 = np.argmin(np.abs(wav - 1.2))
    index_2 = np.argmin(np.abs(wav - 1.65))

    ratio = IF[index_1] / IF[index_2]

    return ratio


def cryst_area(IF, wav):
    def first_degree(w, a, b):
        return a + b * w

    def second_degree(w, a1, b1, c1):
        return a1 * w ** 2 + b1 * w + c1

    wav_fit = []
    IF_fit = []

    index_1 = np.argmin(np.abs(wav - 1.5))
    index_2 = np.argmin(np.abs(wav - 1.61))
    index_3 = np.argmin(np.abs(wav - 1.75))

    wav_fit.append(wav[index_1])
    wav_fit.append(wav[index_2])
    wav_fit.append(wav[index_3])

    IF_fit.append(IF[index_1])
    IF_fit.append(IF[index_2])
    IF_fit.append(IF[index_3])

    fit, covariance = curve_fit(first_degree, wav_fit, IF_fit)

    min_w = 1.5
    max_w = 1.75

    wav_new = []
    IF_new = []

    for i in range(len(wav)):
        if max(wav[0], min_w) < wav[i] < min(wav[-1], max_w):
            wav_new.append(wav[i])
            IF_new.append(IF[i])

    y_linear_fit = first_degree(np.array(wav_new), fit[0], fit[1])

    area = simps(y_linear_fit, wav_new) - simps(IF_new, wav_new)

    wav_2 = []
    IF_2 = []

    for i in range(len(wav)):
        if 1.95 < wav[i] < 2.1:
            wav_2.append(wav[i])
            IF_2.append(IF[i])

    def gaussian(x, A, mu, sigma, baseline):
        return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + baseline

    try:

        wav_2 = []
        IF_2 = []

        for i in range(len(wav)):
            if 1.91 < wav[i] < 2.14:
                wav_2.append(wav[i])
                IF_2.append(IF[i])

        params, covariance = curve_fit(gaussian, wav_2, IF_2, p0=[-0.6, 2, 0.1, 1],maxfev=10000)

        # Extract the fitted parameters
        A_fit, mu_fit, sigma_fit, baseline_fit = params

        # Calculate the error for the fit
        fit_errors = np.sqrt(np.diag(covariance))
        A_error, mu_error, sigma_error, baseline_error = fit_errors

    except RuntimeError:
    # Handle the exception when a RuntimeError occurs
        print("Error: Expanding Fit area to 1.91-2.14")

        wav_2 = []
        IF_2 = []

        for i in range(len(wav)):
            if 1.91 < wav[i] < 2.14:
                wav_2.append(wav[i])
                IF_2.append(IF[i])

        params, covariance = curve_fit(gaussian, wav_2, IF_2, p0=[-0.6, 2, 0.1, 1], maxfev=10000)

        # Extract the fitted parameters
        A_fit, mu_fit, sigma_fit, baseline_fit = params

        # Calculate the error for the fit
        fit_errors = np.sqrt(np.diag(covariance))
        A_error, mu_error, sigma_error, baseline_error = fit_errors

    opt = {'fit': fit, 'area': area, 'loc_2': [mu_fit, mu_error], 'fit2': params}

    return opt


def cryst_area_one_peak(IF, wav):
    def first_degree(w, a, b):
        return a + b * w

    wav_fit = []
    IF_fit = []

    index_2 = np.argmin(np.abs(wav - 1.61))
    index_3 = np.argmin(np.abs(wav - 1.75))

    wav_fit.append(wav[index_2])
    wav_fit.append(wav[index_3])

    IF_fit.append(IF[index_2])
    IF_fit.append(IF[index_3])

    fit, covariance = curve_fit(first_degree, wav_fit, IF_fit)

    min_w = 1.61
    max_w = 1.75

    wav_new = []
    IF_new = []

    for i in range(len(wav)):
        if max(wav[0], min_w) < wav[i] < min(wav[-1], max_w):
            wav_new.append(wav[i])
            IF_new.append(IF[i])

    y_linear_fit = first_degree(np.array(wav_new), fit[0], fit[1])

    area = simps(y_linear_fit, wav_new) - simps(IF_new, wav_new)

    opt = {'fit': fit, 'area': area}

    return opt


def array_mean(array, error=None):
    arr = np.array(array)

    if error is None:
        n = len(array)
        mean = np.sum(arr) / n
        sample_std_dev = np.sqrt(np.sum((arr - mean) ** 2) / (n - 1))
        mean_error = sample_std_dev / np.sqrt(n)
    else:
        error = np.array(error)
        mean = np.sum(arr / error ** 2) / np.sum(1 / error ** 2)
        mean_error = 1 / np.sqrt(np.sum(1 / error ** 2))

    return [mean, mean_error]


def print_error_correlation(optimized_parameters):
    # FUNCTION THAT PRINTS THE CALCULATED ERRORS AND COVARIANCE MATRIX OF A FIT

    optimized_values = optimized_parameters.x
    # Retrieve the covariance matrix
    cov_matrix = np.linalg.inv(optimized_parameters.jac.T @ optimized_parameters.jac)

    # Calculate the standard errors of the optimized parameters
    parameter_errors = np.sqrt(np.diag(cov_matrix))

    # Print the optimized parameter values and their errors
    # for i, value in enumerate(optimized_values):
    # print(f"Parameter {i + 1}: {value:} +/- {parameter_errors[i]:.6f}")

    # print('')
    # print('Cost = ' + str(optimized_parameters.cost))
    # print('')
    # Calculate correlation matrix
    correlation_matrix = cov_matrix / np.outer(parameter_errors, parameter_errors)

    print("Correlation matrix:")
    print(correlation_matrix)
    #print("Eigenvalues:")
    #print(np.linalg.eigvals(correlation_matrix))

    return parameter_errors


def read_cube(file_path, root, download=False):
    try:
        with open(file_path, 'r') as file:
            # Read the entire file content into a single string
            file_contents = file.read()

            # Split the string by spaces to create a list of values
            ids = file_contents.split()

    except FileNotFoundError:
        print(f"The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    cube = {}
    for id in ids:
        cube[id] = VIMS(id, root=root, download=download)

    return cube


# DOWNLOAD THE CUBE AND ALLOCATE IT TO THE RIGHT FOLDER

def retrieve_cube(cube_id):
    directory = 'C:/Users/USUARIO/Desktop/MSc Thesis/Phase A - Data Analysis/Data/'
    cube = VIMS(cube_id, root=directory)

    origin = directory + cube.fname
    destination = directory + cube.target_name + '/' + str(cube.flyby)

    # If the folder does not exist, create it

    if not os.path.exists(destination):
        os.mkdir(destination)
        print("New folder created")

    if os.path.exists(destination + '/' + cube.fname):
        print("Existing Cube")
    else:
        shutil.move(origin, destination)

    print("Cube saved in: " + destination)

    return cube


def opticalconstants(T, max_wavelength=5.1, min_wavelength=0.35, crystallinity=True, sensitivity=1):
    if crystallinity:
        data = np.loadtxt('./Optical Constants/Crystalline_' + str(T) + '.txt')
    else:
        data = np.loadtxt('./Optical Constants/Amorphous_' + str(T) + '.txt')

    wavelength = data[:, 0]
    n = data[:, 1]
    k = data[:, 2]

    min_i = np.argmax(wavelength > min_wavelength)
    max_i = np.argmin(wavelength < max_wavelength)
    wavelength_range = wavelength[min_i:max_i:sensitivity]
    n_range = n[min_i:max_i:sensitivity]
    k_range = k[min_i:max_i:sensitivity]

    opt = {'wav': wavelength_range, 'n': n_range, 'k': k_range}

    return opt


def inter_optical_constants(wav_c, wav_am, n_c, k_c):
    # wav_new = np.clip(np.array(wav_c), np.array(wav_am).min(), np.array(wav_am).max())

    interp_func_n = interp1d(wav_c, n_c, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_func_k = interp1d(wav_c, k_c, kind='linear', bounds_error=False, fill_value='extrapolate')

    # Interpolate y2 values at x_new
    interpolated_n_c = interp_func_n(wav_am)
    interpolated_k_c = interp_func_k(wav_am)

    opt = {'wav': np.array(wav_am), 'n': interpolated_n_c, 'k': interpolated_k_c}

    return opt

###############################################################################
###############################################################################
###############################################################################
####################### INTIMATE MIXED FUNCTIONS ##############################
###############################################################################
###############################################################################
###############################################################################

def singlescatteringalbedomixed(massfraction_am, w_c, w_am):
    """
    :param massfraction_am: [0,1]
    :param w_c: array, function of wavelength
    :param w_am: array, function of wavelength
    :return: array, function of wavelength
    """

    w_mix = (massfraction_am * np.array(w_am) + (1 - massfraction_am) * np.array(w_c))

    return w_mix


# PHASE FUNCTION MIXED WITH WAVELENGTH DEPENDENCE; BEFORE IT WAS JUST A SCALAR
def phasefunctionmixed(massfraction_am, w_c, w_am, phase_c, phase_am):
    """
    :param massfraction_am: [0,1]
    :param w_c: array, function of wavelength
    :param w_am: array, function of wavelength
    :param phase_c: double
    :param phase_am: double
    :return: phase function now becomes function of wavelength
    """

    phase_mix = np.empty((len(w_c)))
    for i in range(len(w_c)):
        phase_mix[i] = (massfraction_am * w_am[i] * phase_am + (1 - massfraction_am) * w_c[i] * phase_c) / (
                massfraction_am * w_am[i] + (1 - massfraction_am) * w_c[i])

    return phase_mix


def shoe_amp_mix(massfraction_am, w_c, w_am, phase_c, phase_am, S_c, S_am):
    b_s0 = np.empty(len(w_c))

    for i in range(len(w_c)):
        b_s0[i] = (massfraction_am * S_am[i] + (1 - massfraction_am) * S_c[i]) / (
                massfraction_am * w_am[i] * phase_am + (1 - massfraction_am) * w_c[i] * phase_c)

    return b_s0

def shoe_amp(w, phase, S_0):
    b_s0 = np.empty(len(w))

    for i in range(len(w)):
        b_s0[i] = S_0[i] / (w[i] * phase)

    return b_s0

###############################################################################
###############################################################################
###############################################################################
############################# FIT FUNCTIONS ###################################
###############################################################################
###############################################################################
###############################################################################

def random_fit(n_fits, hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am):
    fit_D = []
    fit_theta = []
    cost = []
    param_aux = {}
    bounds = ([0,0.000001], [np.deg2rad(90),0.001], )

    fits = {}

    for i in range(n_fits):
        # print('Fit: ' + str(i) + '/' + str(n_fits))

        param = [random.uniform(0.0001, np.deg2rad(90)), random.uniform(0.000001, 0.001)]
        param_aux[i] = param
        fits[i] = optimize.least_squares(
            cost_function_mixed, param,
            args=(hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am, 0, 2.75), bounds=bounds,
        )

        fit_theta.append(fits[i].x[0])
        fit_D.append(fits[i].x[1])
        cost.append(fits[i].cost)

    dif = [max(fit_theta) - min(fit_theta), max(fit_D) - min(fit_D), max(cost) - min(cost)]
    dif_theta = (max(fit_theta) - min(fit_theta)) / max(fit_theta)
    dif_D = (max(fit_D) - min(fit_D)) / max(fit_D)
    dif_cost = (max(cost) - min(cost)) / max(cost)

    clusters = 1  # Initialize with the first cluster.
    sorted_cost = sorted(cost)

    for i in range(1, len(sorted_cost)):
        if (sorted_cost[i] - sorted_cost[i - 1]) / sorted_cost[0] > 0.01:
            clusters += 1

    index_of_lowest_value = cost.index(min(cost))
    print(sorted_cost)
    print(dif)
    if dif_theta < 0.01 and dif_D < 0.01 and dif_cost < 0.01:
        return {'dif': dif, 'fit': fits[0], 'n_sol': clusters, 'Unique': True}
    else:
        return {'dif': dif, 'fit': fits[index_of_lowest_value], 'n_sol': clusters, 'Unique': False}

def random_fit_b(n_fits, hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am,b):
    fit_D = []
    fit_theta = []
    cost = []
    param_aux = {}
    bounds = ([0,0.000001], [np.deg2rad(90),0.001], )

    fits = {}

    for i in range(n_fits):
        # print('Fit: ' + str(i) + '/' + str(n_fits))

        param = [random.uniform(0.0001, np.deg2rad(90)), random.uniform(0.000001, 0.001)]
        param_aux[i] = param
        fits[i] = optimize.least_squares(
            cost_function_mixed_b, param,
            args=(hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am,b, 0, 2.75), bounds=bounds,
        )

        fit_theta.append(fits[i].x[0])
        fit_D.append(fits[i].x[1])
        cost.append(fits[i].cost)

    dif = [max(fit_theta) - min(fit_theta), max(fit_D) - min(fit_D), max(cost) - min(cost)]
    dif_theta = (max(fit_theta) - min(fit_theta)) / max(fit_theta)
    dif_D = (max(fit_D) - min(fit_D)) / max(fit_D)
    dif_cost = (max(cost) - min(cost)) / max(cost)

    clusters = 1  # Initialize with the first cluster.
    sorted_cost = sorted(cost)

    for i in range(1, len(sorted_cost)):
        if (sorted_cost[i] - sorted_cost[i - 1]) / sorted_cost[0] > 0.01:
            clusters += 1

    index_of_lowest_value = cost.index(min(cost))
    print(sorted_cost)
    print(dif)
    if dif_theta < 0.01 and dif_D < 0.01 and dif_cost < 0.01:
        return {'dif': dif, 'fit': fits[0], 'n_sol': clusters, 'Unique': True}
    else:
        return {'dif': dif, 'fit': fits[index_of_lowest_value], 'n_sol': clusters, 'Unique': False}


def random_fit_phi(n_fits, hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am,phi):
    fit_D = []
    fit_theta = []
    cost = []
    param_aux = {}
    bounds = ([0,0.000001], [np.deg2rad(90),0.001], )

    fits = {}

    for i in range(n_fits):
        # print('Fit: ' + str(i) + '/' + str(n_fits))

        param = [random.uniform(0.0001, np.deg2rad(90)), random.uniform(0.000001, 0.001)]
        param_aux[i] = param
        fits[i] = optimize.least_squares(
            cost_function_mixed_phi, param,
            args=(hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am,phi, 0, 2.75), bounds=bounds,
        )

        fit_theta.append(fits[i].x[0])
        fit_D.append(fits[i].x[1])
        cost.append(fits[i].cost)

    dif = [max(fit_theta) - min(fit_theta), max(fit_D) - min(fit_D), max(cost) - min(cost)]
    dif_theta = (max(fit_theta) - min(fit_theta)) / max(fit_theta)
    dif_D = (max(fit_D) - min(fit_D)) / max(fit_D)
    dif_cost = (max(cost) - min(cost)) / max(cost)

    clusters = 1  # Initialize with the first cluster.
    sorted_cost = sorted(cost)

    for i in range(1, len(sorted_cost)):
        if (sorted_cost[i] - sorted_cost[i - 1]) / sorted_cost[0] > 0.01:
            clusters += 1

    index_of_lowest_value = cost.index(min(cost))
    print(sorted_cost)
    print(dif)
    if dif_theta < 0.01 and dif_D < 0.01 and dif_cost < 0.01:
        return {'dif': dif, 'fit': fits[0], 'n_sol': clusters, 'Unique': True}
    else:
        return {'dif': dif, 'fit': fits[index_of_lowest_value], 'n_sol': clusters, 'Unique': False}


def random_fit_norm(n_fits, hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am):
    fit_D = []
    fit_theta = []
    cost = []
    param_aux = {}
    bounds = ([0,0,0,0.000001], [1,np.deg2rad(45),0.74,0.001], )

    fits = {}

    for i in range(n_fits):
        # print('Fit: ' + str(i) + '/' + str(n_fits))
        param = [random.uniform(0.0001, 0.999), random.uniform(0.0001, np.deg2rad(45)), random.uniform(0.0001, 0.74),
                 random.uniform(0.000001, 0.001)]
        param_aux[i] = param
        fits[i] = optimize.least_squares(
            cost_function_mixed_no_weight, param,
            args=(hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am, 0, 2.75), bounds=bounds,
        )

        fit_theta.append(fits[i].x[0])
        fit_D.append(fits[i].x[1])

        cost.append(fits[i].cost)

    dif = [max(fit_theta) - min(fit_theta), max(fit_D) - min(fit_D), max(cost) - min(cost)]
    dif_theta = (max(fit_theta) - min(fit_theta)) / max(fit_theta)
    dif_D = (max(fit_D) - min(fit_D)) / max(fit_D)
    dif_cost = (max(cost) - min(cost)) / max(cost)

    clusters = 1  # Initialize with the first cluster.
    sorted_cost = sorted(cost)

    for i in range(1, len(sorted_cost)):
        if (sorted_cost[i] - sorted_cost[i - 1]) / sorted_cost[0] > 0.01:
            clusters += 1

    index_of_lowest_value = cost.index(min(cost))
    print(sorted_cost)
    print(dif)
    if dif_theta < 0.01 and dif_D < 0.01 and dif_cost < 0.01:
        return {'dif': dif, 'fit': fits[0], 'n_sol': clusters, 'Unique': True}
    else:
        return {'dif': dif, 'fit': fits[index_of_lowest_value], 'n_sol': clusters, 'Unique': False}


def cost_function_mixed_no_weight(parameters, hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am,
                                  min_w=0, max_w=6, body='0'):
    IF_hapke = hapke_model_mixed(parameters, hapke_wav, angles, n_c, k_c, n_am, k_am, body)['IF']

    interp_func_1 = interp1d(hapke_wav, IF_hapke, bounds_error=False)
    interp_func_2 = interp1d(measured_wav, measured_IF, bounds_error=False)

    wav_new = []

    for i in range(len(measured_wav)):
        if max(measured_wav[0], hapke_wav[0], min_w) < measured_wav[i] < min(measured_wav[-1], hapke_wav[-1], max_w):
            wav_new.append(measured_wav[i])

    interpolated_hapke = interp_func_1(wav_new)
    interpolated_lab = interp_func_2(wav_new)

    difference = (interpolated_hapke - interpolated_lab)

    return difference

def cost_function_mixed(parameters, hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am, min_wav=0,
                        max_w=6):
    IF_hapke = hapke_model_mixed(parameters, hapke_wav, angles, n_c, k_c, n_am, k_am)['IF']

    interp_func_1 = interp1d(hapke_wav, IF_hapke, bounds_error=False)
    interp_func_2 = interp1d(measured_wav, measured_IF, bounds_error=False)

    wav_new = []

    for i in range(len(measured_wav)):
        if max(measured_wav[0], hapke_wav[0], min_wav) < measured_wav[i] < min(measured_wav[-1], hapke_wav[-1], max_w):
            wav_new.append(measured_wav[i])

    interpolated_hapke = interp_func_1(wav_new)
    interpolated_lab = interp_func_2(wav_new)

    difference = (interpolated_hapke - interpolated_lab) / interpolated_lab
    #plot_param['cost'].append(np.linalg.norm(difference))
    #plot_param['D'].append(parameters[1])
    #plot_param['theta'].append(parameters[0])
    #print(np.linalg.norm(difference))
    return difference

def cost_function_mixed_mass_fraction(parameters, hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am,
                                      aux_param, min_w=1.57, max_w=1.7, normalized_wav=0):
    IF_hapke = hapke_model_mixed_mass_fraction(parameters, hapke_wav, angles, n_c, k_c, n_am, k_am, aux_param)['IF']

    interp_func_1 = interp1d(hapke_wav, IF_hapke, bounds_error=False)
    interp_func_2 = interp1d(measured_wav, measured_IF, bounds_error=False)

    wav_new = []

    for i in range(len(measured_wav)):
        if max(measured_wav[0], hapke_wav[0], min_w) < measured_wav[i] < min(measured_wav[-1], hapke_wav[-1], max_w):
            wav_new.append(measured_wav[i])

    if normalized_wav == 0:
        interpolated_hapke = interp_func_1(wav_new)
        interpolated_lab = interp_func_2(wav_new)
    else:
        idx = (np.abs(np.array(wav_new) - normalized_wav)).argmin()

        interpolated_hapke = interp_func_1(wav_new)
        interpolated_hapke = interpolated_hapke / interpolated_hapke[idx]
        interpolated_lab = interp_func_2(wav_new)
        interpolated_lab = interpolated_lab / interpolated_lab[idx]

    difference = (interpolated_hapke - interpolated_lab) / interpolated_lab

    return difference


###############################################################################
###############################################################################
###############################################################################
############################# PLOTTING FUNCTIONS ###################################
###############################################################################
###############################################################################
###############################################################################

def plot_pixel_equi(wav, cube, pixel_loc, background=False):
    pixel = {}
    patches = []

    for i in range(len(pixel_loc)):
        pixel[i] = cube @ pixel_loc[i]

        corners_lon = pixel[i].corners.lonlat[0, :]
        corners_lat = pixel[i].corners.lonlat[1, :]

        for j in range(len(corners_lon)):
            if corners_lon[j] > 180:
                corners_lon[j] += -360

        vertices = [(corners_lon[0], corners_lat[0]), (corners_lon[1], corners_lat[1]),
                    (corners_lon[2], corners_lat[2]),
                    (corners_lon[3], corners_lat[3])]

        square_patch = Polygon(vertices, closed=True, color='yellow', alpha=0.7)
        patches.append(square_patch)

    p = PatchCollection(patches)

    if background:

        bg = MAPS[cube.target_name]
        fig, ax = bg.figure(figsize=(20, 10))

    else:
        ax = cube.plot(wav, 'equi')

    ax.add_collection(p)

    return ax


def generate_patches(patches, pixel):
    corners_lon = pixel.corners.lonlat[0, :]
    corners_lat = pixel.corners.lonlat[1, :]

    for j in range(len(corners_lon)):
        if corners_lon[j] > 180:
            corners_lon[j] += -360

    vertices = [(corners_lon[0], corners_lat[0]), (corners_lon[1], corners_lat[1]),
                (corners_lon[2], corners_lat[2]),
                (corners_lon[3], corners_lat[3])]

    square_patch = Polygon(vertices, closed=True, alpha=0.7)

    patches.append(square_patch)


def is_micron_dip(pixel):
    index = np.argmin(np.abs(pixel.wvlns - 1.25))

    slope = pixel.spectrum[index] / pixel.spectrum[index - 1]

    if slope < 0.9:
        aux = False
    else:
        aux = True

    return aux
