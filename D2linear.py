import numpy as np
from numpy.fft import fft2, ifft2
from numpy import pi, abs, fabs, sign, sqrt

def regularize(x):
    eps = 1e-18
    if fabs(x) < 1e-18:
        if fabs(x) >= 0:
            return eps
        else:
            return -eps
    else:
        return x

@profile
def D2linear(topo, u0, v0, tau, T0, Nm):
    ny, nx = topo.shape

    delx = 750.0
    dely = 750.0

    tauc = tau / 2.0
    tauf = tau / 2.0

    Hw = 2500.0
    Cw = 0.008293103448275862

    # pre-compute kx
    dkx = 2.0 * pi / (nx * delx)
    kx = np.zeros(nx)
    nchx = (nx / 2) + 1
    for i in xrange(0, nchx):
        kx[i] = i * dkx
    for i in xrange(nchx, nx):
        kx[i] = - (nx - i) * dkx

    # pre-compute ky
    dky = 2.0 * pi / (ny * dely)
    ky = np.zeros(ny)
    nchy = (ny / 2) + 1
    for i in xrange(0, nchy):
        ky[i] = i * dky
    for i in xrange(nchy, ny):
        ky[i] = - (ny - i) * dky

    h_hat = fft2(topo)
    P_hat = np.zeros_like(h_hat)

    for j in xrange(ny):
        for i in xrange(nx):

            sigma = u0 * kx[i] + v0 * ky[j]

            C = regularize(sigma**2.0)

            m_squared = ((Nm**2 - sigma**2) / C) * (kx[i]**2 + ky[j]**2)

            if m_squared >= 0.0:
                if sigma == 0.0:
                    m = sqrt(m_squared)
                else:
                    m = sign(sigma) * sqrt(m_squared)
            else:
                m = sqrt(-1 * m_squared)

            assert np.abs(1 - 1j * m * Hw) > 1e-10

            # see equation 49
            D = (1.0 - 1j * m * Hw) * (1.0 + 1j * sigma * tauc) * (1.0 + 1j * sigma * tauf)
            P_hat[j, i] = Cw * 1j * sigma * h_hat[j, i] / D

    P = np.real(ifft2(P_hat)) * 3600.0
    P = np.maximum(P, 0.0)

    return P
