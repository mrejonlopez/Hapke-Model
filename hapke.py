import numpy as np


# SINGLE SCATTERING ALBEDO FUNCTION

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

    R_0 = np.empty((len(n)))
    Se = np.empty((len(n)))
    Si = np.empty((len(n)))
    alpha = np.empty((len(n)))
    Dmodif = np.empty((len(n)))
    r_i = np.empty((len(n)))
    Theta = np.empty((len(n)))
    w = np.empty((len(n)))

    for i in range(len(n)):
        R_0[i] = ((n[i] - 1) ** 2 + k[i] ** 2) / ((n[i] + 1) ** 2 + k[i] ** 2)

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
    phi, D, theta_bar = parameters
    eme, inc, phase = angles
    # Calculate the modeled spectrum using Hapke model equations

    psi = phasetoazimuth(phase, eme, inc)

    B_S0 = 0.53
    b = 0.2
    B_C0 = 0.35

    transport_mean_free_path = 33 * 10 ** (-6)

    K = porosityparameter(phi)

    B_SH = shoe(B_S0, phi, phase)
    c = hockey_stick(b)

    p = phase_function(b, c, phase)
    [mu_0e, mu_e, S] = shadowingfunction(inc, eme, psi, theta_bar)

    w = singlescatteringalbedo(n_range, k_range, wav, D)
    B_CB = cboe(B_C0, transport_mean_free_path, wav, phase, K)

    r = []
    for i in range(len(wav)):
        r.append(K * w[i] / (4 * np.pi) * mu_0e / (mu_0e + mu_e) * (
                p * B_SH + H(w[i], mu_0e) * H(w[i], mu_e) - 1) * B_CB[i] * S)

    IF = np.array(r) * np.pi

    output = {'r': r, 'IF': IF, 'w': w, 'p': p}

    return output


def singlescatteringalbedomixed(massfraction_am, w_c, w_am):
    """
    :param massfraction_am: [0,1]
    :param w_c: array, function of wavelength
    :param w_am: array, function of wavelength
    :return: array, function of wavelength
    """

    w_mix = massfraction_am * np.array(w_am) + (1 - massfraction_am) * np.array(w_c)

    return w_mix


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
        return np.exp(-2 / (np.pi * np.tan(thetabar) * np.tan(x)))

    def E2(x):
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
        S = (mu_e / eta(e)) * (mu_0 / eta(i)) * (xi / (1 - fpsi(psi) + fpsi(psi) * xi * (mu_0 / eta(e))))

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
    B_S = (1 + np.tan(g / 2) / h_s)

    return 1 + B_S0 / B_S


print("listo")


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
