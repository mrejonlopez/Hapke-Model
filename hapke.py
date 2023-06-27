import numpy as np
from scipy.interpolate import interp1d

density_cr = 0.931
density_am = 0.94


# DECIDED TO TAKE THE SAME DENSITY AS THE DIFFERENCE IS VERY SMALL

###############################################################################
###############################################################################
###############################################################################
####################### MAIN HAPKE FUNCTIONS ##################################
###############################################################################
###############################################################################
###############################################################################

def hapke_model_mixed(parameters, theta_bar, wav, angles, n_c, k_c, n_am, k_am):
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
    phi, D, b, mass_fraction = parameters

    eme, inc, phase = angles
    # Calculate the modeled spectrum using Hapke model equations

    psi = phasetoazimuth(phase, eme, inc)

    w_c = singlescatteringalbedo(n_c, k_c, wav, D)
    w_am = singlescatteringalbedo(n_am, k_am, wav, D)

    b = 0.2
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

def hapke_model_mixed_no_shoe(parameters, wav, angles, n_c, k_c, n_am, k_am):
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
    phi, D, mass_fraction = parameters

    theta_bar = np.deg2rad(20)

    eme, inc, phase = angles
    # Calculate the modeled spectrum using Hapke model equations

    psi = phasetoazimuth(phase, eme, inc)

    w_c = singlescatteringalbedo(n_c, k_c, wav, D)
    w_am = singlescatteringalbedo(n_am, k_am, wav, D)

    b = 0.2
    c = hockey_stick(b)
    p = phase_function(b, c, phase)

    #R0_c = fresnel_coefficient(n_c, k_c)
    #R0_am = fresnel_coefficient(n_am, k_am)

    #B_S0 = shoe_amp_mix(mass_fraction, wav, wav, p, p, R0_c, R0_am)

    K = porosityparameter(phi)

    #B_SH = shoe(B_S0, phi, phase)

    [mu_0e, mu_e, S] = shadowingfunction(inc, eme, psi, theta_bar)

    w = singlescatteringalbedomixed(mass_fraction, w_c, w_am)

    r = np.empty(len(wav))

    for i in range(len(wav)):
        r[i] = (K * w[i] / (4 * np.pi) * mu_0e / (mu_0e + mu_e) * (
                p + H(w[i], mu_0e) * H(w[i], mu_e) - 1) * S)

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
    phi, D, b = parameters
    eme, inc, phase = angles
    # Calculate the modeled spectrum using Hapke model equations
    theta_bar = 0
    psi = phasetoazimuth(phase, eme, inc)

    B_S0 = 0.5
    #b = 0.2

    K = porosityparameter(phi)

    B_SH = shoe(B_S0, phi, phase)
    c = hockey_stick(b)

    p = phase_function(b, c, phase)

    [mu_0e, mu_e, S] = shadowingfunction(inc, eme, psi, theta_bar)

    w = singlescatteringalbedo(n_range, k_range, wav, D)

    r = []
    for i in range(len(wav)):
        r.append(K * w[i] / (4 * np.pi) * mu_0e / (mu_0e + mu_e) * (
                p * B_SH + H(w[i], mu_0e) * H(w[i], mu_e) - 1) * S)

    IF = np.array(r) * np.pi

    output = {'r': r, 'IF': IF, 'w': w, 'p': p}

    return output


def hapke_model2(parameters, wav, angles, n, k):

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
    phi, D, theta_bar = parameters
    eme, inc, phase = angles
    # Calculate the modeled spectrum using Hapke model equations

    psi = phasetoazimuth(phase, eme, inc)

    b = 0.2

    K = porosityparameter(phi)
    c = hockey_stick(b)

    p = phase_function(b, c, phase)
    w = singlescatteringalbedo(n, k, wav, D)

    R_0 = np.empty(len(wav))
    B_S0 = np.empty(len(wav))

    for i in range(len(B_S0)):
        R_0[i] = ((n[i] - 1) ** 2 + k[i] ** 2) / ((n[i] + 1) ** 2 + k[i] ** 2)

        B_S0[i] = R_0[i] / (w[i] * phase)

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


def opticalconstants(T, max_wavelength=5.1, min_wavelength=0.35, crystallinity=True):
    if crystallinity:
        data = np.loadtxt('./Optical Constants/Crystalline_' + str(T) + '.txt')
    else:
        data = np.loadtxt('./Optical Constants/Amorphous_' + str(T) + '.txt')

    wavelength = [row[0] for row in data]
    n = [row[1] for row in data]
    k = [row[2] for row in data]

    indices = [i for i in range(len(wavelength)) if min_wavelength <= wavelength[i] <= max_wavelength]

    # Extract wavelength, n, and k values within desired range
    wavelength_range = [wavelength[i] for i in indices]
    n_range = [n[i] for i in indices]
    k_range = [k[i] for i in indices]

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


###############################################################################
###############################################################################
###############################################################################
############################# FIT FUNCTIONS ###################################
###############################################################################
###############################################################################
###############################################################################

def cost_function(parameters, hapke_wav, angles, measured_IF, measured_wav, n_range, k_range, max_w=6):

    IF_hapke = hapke_model(parameters, hapke_wav, angles, n_range, k_range)['IF']

    interp_func_1 = interp1d(hapke_wav, IF_hapke, kind='cubic', bounds_error=False)
    interp_func_2 = interp1d(measured_wav, measured_IF, kind='cubic', bounds_error=False)

    wav_new = []

    for i in range(len(measured_wav)):
        if max(measured_wav[0], hapke_wav[0]) < measured_wav[i] < min(measured_wav[-1], hapke_wav[-1], max_w):
            wav_new.append(measured_wav[i])

    interpolated_hapke = interp_func_1(wav_new)
    interpolated_lab = interp_func_2(wav_new)

    difference = interpolated_hapke - interpolated_lab

    return difference


#def cost_function(parameters, hapke_wav, angles, measured_IF, measured_wav, n_range, k_range, max_w=0):
#
#    IF_hapke = hapke_model(parameters, hapke_wav, angles, n_range, k_range)['IF']
#
#    if max_w == 0:
#        wav_new = np.unique(np.array(measured_wav), np.array(hapke_wav).min(), np.array(hapke_wav).max())
#    else:
#        wav_new = np.unique(np.array(measured_wav), np.array(hapke_wav).min(), max_w)
#    #print(wav_new)
#    print(measured_wav)
#    print(hapke_wav)
#
#    interp_func = interp1d(measured_wav, measured_IF, kind='linear', bounds_error=False, fill_value='extrapolate')
#
#    # Interpolate y2 values at x_new
#    interpolated_IF = interp_func_n(hapke_wav)
#    #interpolated_k_c = interp_func_k(wav_am)
#
#
#    #interp_func = interp1d(hapke_wav, IF_hapke, kind='linear')
#
#    # Interpolate y2 values at x_new
#    #interpolated_hapke = interp_func(wav_new)
#
#
#    difference = interpolated_IF - IF_hapke
#
#    return difference

def cost_function_mixed(parameters, hapke_wav, angles, measured_IF, measured_wav, n1, k1, n2, k2, max_w=0):
    IF_hapke = hapke_model_mixed_no_shoe(parameters, hapke_wav, angles, n1, k1, n2, k2)['IF']

    if max_w == 0:
        wav_new = np.clip(np.array(measured_wav), np.array(hapke_wav).min(), np.array(hapke_wav).max())
    else:
        wav_new = np.clip(np.array(measured_wav), np.array(hapke_wav).min(), max_w)

    interp_func = interp1d(hapke_wav, IF_hapke, kind='linear')

    # Interpolate y2 values at x_new
    interpolated_hapke = interp_func(wav_new)

    difference = interpolated_hapke - measured_IF

    return difference

def cost_function_r(parameters, theta_bar, hapke_wav, angles, measured_IF, measured_wav, n1, k1, n2, k2, max_w=0):

    IF_hapke = hapke.hapke_model(ini_par, wav, angles, n, k)['IF']

    interp_func_1 = interp1d(wav, IF_hapke, kind='cubic', bounds_error=False)
    interp_func_2 = interp1d(wav_lab, r_lab, kind='cubic', bounds_error=False)

    wav_new = []
    max_w = 5

    for i in range(len(wav_lab)):
        if max(wav_lab[0], wav[0]) < wav_lab[i] < min(wav_lab[-1], wav[-1], max_w):
            wav_new.append(wav_lab[i])

    interpolated_IF = interp_func_1(wav_new)
    new_r_lab = interp_func_2(wav_new)

    difference = interpolated_IF - new_r_lab

    return difference

class icymoons:
    def __init__(self, B_C0, freepath, B_S0, b):
        self.b_co = B_C0
        self.freepath = freepath
        self.B_S0 = B_S0
        self.b = b


# DEFINITION OF THE ATTRIBUTES OF THE ICY MOONS - BOOK ENCELADUS AND THE ICY MOONS: SURFACE PROPERTIES

MIMAS = icymoons(0.31, 19 * 10 ** (-6), 0.53, 0.175)
ENCELADUS = icymoons(0.35, 33 * 10 ** (-6), 0.53, 0.1)
TETHYS = icymoons(0.32, 109 * 10 ** (-6), 0.53, 0.25)
DIONE = icymoons(0.32, 11 * 10 ** (-6), 0.53, 0.2)  # UNDETERMINED
RHEA = icymoons(0.33, 31 * 10 ** (-6), 0.53, 0.45)  # AVERAGE OF 3 VALUES
IAPETUS = icymoons(0.35, 33 * 10 ** (-6), 0.53, 0.2)  # UNDETERMINED
