import numpy as np
from matplotlib.collections import PatchCollection
from scipy.interpolate import interp1d
import os
from pyvims import VIMS
import shutil
from matplotlib.patches import Polygon
from pyvims.misc import MAPS
from scipy.optimize import curve_fit
from scipy.integrate import simps

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

    if body == 'MIMAS':
        theta_bar = MIMAS.theta_bar
        b = MIMAS.b
    elif body == 'ENCELADUS':
        theta_bar = ENCELADUS.theta_bar
        b = ENCELADUS.b
    elif body == 'RHEA':
        phi = RHEA.theta_bar
        b = RHEA.b
    elif body == 'IAPETUS':
        theta_bar = IAPETUS.theta_bar
        b = IAPETUS.b
    elif body == 'TETHYS':
        theta_bar = TETHYS.theta_bar
        b = TETHYS.b
    elif body == 'DIONE':
        theta_bar = DIONE.theta_bar
        b = DIONE.b
    else:
        #print('NO BODY IDENTIFIED, ARBITRARY VALUES OF b AND ROUGHNESS')
        unbarquitodevela = 1
        #theta_bar = np.deg2rad(5)
        #b = 0.2
    # Unpack the parameters
    # Assuming you have parameters p1, p2, p3, etc.

    mass_fraction = 0
    b = 0.25
    phi = 0.15
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

    b = 0.25
    phi = 0.15
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


def hapke_model_mixed_step(parameters, wav, angles, n_c, k_c, n_am, k_am, aux_param, fit_set=0):
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
    if fit_set == 0:
        b, theta_bar = parameters
        phi, D = aux_param
    else:
        b, theta_bar = aux_param
        phi, D = parameters

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

def print_error_correlation(optimized_parameters):
    # FUNCTION THAT PRINTS THE CALCULATED ERRORS AND COVARIANCE MATRIX OF A FIT

    optimized_values = optimized_parameters.x
    # Retrieve the covariance matrix
    cov_matrix = np.linalg.inv(optimized_parameters.jac.T @ optimized_parameters.jac)

    # Calculate the standard errors of the optimized parameters
    parameter_errors = np.sqrt(np.diag(cov_matrix))

    # Print the optimized parameter values and their errors
    for i, value in enumerate(optimized_values):
        print(f"Parameter {i + 1}: {value:} +/- {parameter_errors[i]:.6f}")

    print('')
    print('Cost = ' + str(optimized_parameters.cost))
    print('')
    # Calculate correlation matrix
    correlation_matrix = cov_matrix / np.outer(parameter_errors, parameter_errors)

    print("Correlation matrix:")
    print(correlation_matrix)

    return parameter_errors


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

def cost_function(parameters, hapke_wav, angles, measured_IF, measured_wav, n_range, k_range, max_w=6):
    IF_hapke = hapke_model(parameters, hapke_wav, angles, n_range, k_range)['IF']

    interp_func_1 = interp1d(hapke_wav, IF_hapke, bounds_error=False)
    interp_func_2 = interp1d(measured_wav, measured_IF, bounds_error=False)

    wav_new = []

    for i in range(len(measured_wav)):
        if max(measured_wav[0], hapke_wav[0]) < measured_wav[i] < min(measured_wav[-1], hapke_wav[-1], max_w):
            wav_new.append(measured_wav[i])

    interpolated_hapke = interp_func_1(wav_new)
    interpolated_lab = interp_func_2(wav_new)

    difference = (interpolated_hapke - interpolated_lab) / interpolated_lab

    return difference


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


def cost_function_no_weight(parameters, hapke_wav, angles, measured_IF, measured_wav, n_range, k_range, max_w=6):
    IF_hapke = hapke_model(parameters, hapke_wav, angles, n_range, k_range)['IF']

    interp_func_1 = interp1d(hapke_wav, IF_hapke, bounds_error=False)
    interp_func_2 = interp1d(measured_wav, measured_IF, bounds_error=False)

    wav_new = []

    for i in range(len(measured_wav)):
        if max(measured_wav[0], hapke_wav[0]) < measured_wav[i] < min(measured_wav[-1], hapke_wav[-1], max_w):
            wav_new.append(measured_wav[i])

    interpolated_hapke = interp_func_1(wav_new)
    interpolated_lab = interp_func_2(wav_new)

    difference = (interpolated_hapke - interpolated_lab)

    return difference


def cost_function_mixed(parameters, hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am, min_wav=0,max_w=6):
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

    return difference

def crystallinity_coecient(IF,wav):

    index_1 = np.argmin(np.abs(wav - 1.2))
    index_2 = np.argmin(np.abs(wav - 1.65))

    ratio = IF[index_1] / IF[index_2]

    return ratio

def crystallinity_coecient_2(IF,wav):

    index_1 = np.argmin(np.abs(wav - 1.61))
    index_2 = np.argmin(np.abs(wav - 1.65))

    ratio = IF[index_1] / IF[index_2]

    return ratio

def cryst_area(IF,wav):

    def first_degree(w, a, b):
        return a + b * w

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

    y_linear_fit = first_degree(np.array(wav_new),fit[0],fit[1])

    area = simps(y_linear_fit, wav_new) - simps(IF_new, wav_new)

    opt = {'fit': fit, 'area': area}

    return opt

def cost_function_mixed_step(parameters, hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am, aux_param, fit_set, max_w=6):
    IF_hapke = hapke_model_mixed_step(parameters, hapke_wav, angles, n_c, k_c, n_am, k_am,aux_param,fit_set)['IF']

    interp_func_1 = interp1d(hapke_wav, IF_hapke, bounds_error=False)
    interp_func_2 = interp1d(measured_wav, measured_IF, bounds_error=False)

    wav_new = []

    for i in range(len(measured_wav)):
        if max(measured_wav[0], hapke_wav[0]) < measured_wav[i] < min(measured_wav[-1], hapke_wav[-1], max_w):
            wav_new.append(measured_wav[i])

    interpolated_hapke = interp_func_1(wav_new)
    interpolated_lab = interp_func_2(wav_new)

    difference = (interpolated_hapke - interpolated_lab) / interpolated_lab

    return difference

def cost_function_mixed_mass_fraction(parameters, hapke_wav, angles, measured_IF, measured_wav, n_c, k_c, n_am, k_am, aux_param,min_w=1.57, max_w=1.7, normalized_wav=0):
    IF_hapke = hapke_model_mixed_mass_fraction(parameters, hapke_wav, angles, n_c, k_c, n_am, k_am,aux_param)['IF']

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

def crystallinity_fit(IF, wav, min_w=1.57, max_w=1.7, normalized_wav=0):

    def third_degree(w,a,b,c,d):
        return a + b * w + c * w ** 2 + d * w ** 3

    def first_degree(w,a,b):
        return a + b + w

    wav_new = []
    IF_new = []

    for i in range(len(wav)):
        if max(wav[0], min_w) < wav[i] < min(wav[-1], max_w):
            wav_new.append(wav[i])
            IF_new.append(IF[i])

    third_degree_fit = curve_fit(third_degree, wav, IF)
    first_degree_fit = curve_fit(first_degree, wav, IF)
    difference = (interpolated_hapke - interpolated_lab) / interpolated_lab

    return difference

class icymoons:
    def __init__(self, B_C0, freepath, B_S0, b, theta_bar, T):
        self.b_co = B_C0
        self.freepath = freepath
        self.B_S0 = B_S0
        self.b = b
        self.theta_bar = theta_bar
        self.T = T


# DEFINITION OF THE ATTRIBUTES OF THE ICY MOONS - BOOK ENCELADUS AND THE ICY MOONS: SURFACE PROPERTIES

MIMAS = icymoons(0.31, 19 * 10 ** (-6), 0.53, 0.175, np.deg2rad(30), 80)
ENCELADUS = icymoons(0.35, 33 * 10 ** (-6), 0.53, 0.1, np.deg2rad(21), 80)
TETHYS = icymoons(0.32, 109 * 10 ** (-6), 0.53, 0.25, np.deg2rad(23), 60)
DIONE = icymoons(0.32, 11 * 10 ** (-6), 0.53, 0.2, np.deg2rad(20), 100)  # UNDETERMINED
RHEA = icymoons(0.33, 31 * 10 ** (-6), 0.53, 0.45, np.deg2rad(15), 120)  # AVERAGE OF 3 VALUES
IAPETUS = icymoons(0.35, 33 * 10 ** (-6), 0.53, 0.2, np.deg2rad(20), 120)  # UNDETERMINED


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
