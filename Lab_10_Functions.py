from numpy import exp


def psi_function(x=1, length=1e-8, sigma=1e-10, kappa=1e10):
    """
    Returns the first value of psi at t=0 for all x values
    :param x:
    :param length: length of the box, in meters
    :param sigma:
    :param kappa:
    :return:
    """
    x_0 = length / 2
    num = (x - x_0) ** 2
    den = 2 * (sigma ** 2)
    psi_0 = exp(-(num / den)) * exp(1j * kappa * x)
    return psi_0

