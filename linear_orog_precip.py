# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:47:27 2015

@author: leif
"""


from scipy.fftpack import fft2, fftfreq
import numpy as np
import cmath

class OrographicPrecipitation(object):

    """Calculates orographic precipitation following Smith & Barstad (2004).

    """

    def __init__(self, X, Y, U, V, Orography, physical_constants):
        self.X = X
        self.Y = Y
        self.U = U
        self.V = V
        self.Orography = Orography
        self.physical_constants = physical_constants
        self.dx = np.diff(X)[0,0]
        self.dy = np.diff(Y, axis=0)[0,0]
        self.nx = len(Orography[1,:])
        self.ny = len(Orography)

        self.P = self._compute_precip()
        self.P_units = 'm year-1'
        
    def _compute_precip(self):

        physical_constants = self.physical_constants
        Orography = self.Orography
        Orography_fft = np.fft.fft2(Orography)
        dx = self.dx
        dy = self.dy
        nx = self.nx
        ny = self.ny
        U = self.U
        V = self.V
        
        x_n_value = np.fft.fftfreq(len(Orography[1,:]), (1.0 / len(Orography[1,:])))
        y_n_value = np.fft.fftfreq(len(Orography), (1.0/len(Orography)))

        x_len = len(Orography) * dx
        y_len = len(Orography[1,:]) * dy

        kx_line = np.divide(np.multiply(2.0 * np.pi, x_n_value), x_len)
        ky_line = np.divide(np.multiply(2.0 * np.pi, y_n_value), y_len)[np.newaxis].T

        kx = np.tile(kx_line, (ny, 1))
        ky = np.tile(ky_line, (1, nx))

        # Intrinsic frequency sigma = U*k + V*l
        sigma = np.add(np.multiply(kx, U), np.multiply(ky, V))

        # The vertical wave number
        # Eqn. 38
        # m = np.power(np.multiply(np.add(np.power(kx, 2.) , np.power(ky, 2.)) , np.divide(np.subtract(physical_constants['Nm']**2., np.power(sigma, 2.)) , np.subtract(np.power(sigma, 2.), physical_constants['f']**2.))), 0.5)
        m = np.power(np.multiply(np.divide(np.subtract(physical_constants['Nm']**2, np.power(sigma, 2.)), np.power(sigma, 2.)), np.add(np.power(kx, 2.) , np.power(ky, 2.))), 0.5)
        
        m[np.isnan(m)] = 0
        #m = np.maximum(m,0.00001)
        m = np.minimum(m,0.01)
        
        # Numerator in Eqn. 49
        P_karot_num = np.multiply(np.multiply(np.multiply(physical_constants['Cw'], cmath.sqrt(-1)), sigma), Orography_fft)
        P_karot_denom_Hw = np.subtract(np.multiply(np.multiply(physical_constants['Hw'], m), cmath.sqrt(-1)), 1)
        P_karot_denom_tauc =  np.add(np.multiply(np.multiply(sigma, physical_constants['tau_c']), cmath.sqrt(-1)), 1)
        P_karot_denom_tauf = np.add(np.multiply(np.multiply(sigma, physical_constants['tau_f']), cmath.sqrt(-1)), 1)
        # Denominator in Eqn. 49
        P_karot_denom = np.multiply(P_karot_denom_Hw, np.multiply(P_karot_denom_tauc, P_karot_denom_tauf))
        P_karot = np.divide(P_karot_num, P_karot_denom)      

        P_karot_amp = np.absolute(P_karot)  # get the amplitude
        P_karot_angle = np.angle(P_karot)   # get the phase angle

        # Converting from wave domain back to space domain
        # Eqn. 6
        y2 = np.multiply(P_karot_amp,  np.add(np.cos(P_karot_angle) , np.multiply(cmath.sqrt(-1) , np.sin(P_karot_angle))))
        y3 = np.fft.ifft2(y2)
        spy = 31556925.9747
        P = np.multiply(np.real(y3), spy*1./1000)   # m year-1
        # Truncation
        P[P<0] = 0
        return P



if __name__ == "__main__":
    print('Linear Orographic Precipitation Model by Smith & Barstad (2004)')

    import pylab as plt
    from matplotlib import cm
    import itertools
    from cf_units import Unit
    
    dx = dy = 750.
    x, y = np.arange(-100e3, 200e3, dx), np.arange(-150e3, 150e3, dy)
    nx, nx = len(x), len(y)
    h_max = 500.
    x0 = y0 = 0
    sigma_x = sigma_y = 15e3

    tau_c_values = [1000]
    tau_f_values = [1000]
    Cw_values = [0.0]
    Nm_values = [0.005]
    Hw_values = [2500]
    u_values = [-15]

    Theta_m = -6.5
    rho_Sref = 7.4e-3  # kg m-3
    gamma = -5.8

    Pdata = []
    combinations = list(itertools.product(tau_c_values, tau_f_values, Cw_values, Nm_values, Hw_values, u_values))
    for combination in combinations:
            
        tau_c, tau_f, Cw, Nm, Hw, u = combination
        physical_constants = dict()
        physical_constants['tau_c'] = tau_c  # conversion time [s]
        physical_constants['tau_f'] = tau_f  # fallout time [s]
        physical_constants['f'] = 2 * 7.2921e-5 * np.sin(60 * np.pi / 180)
        physical_constants['Nm'] = Nm        # moist stability frequency [s-1]
        physical_constants['Cw'] = rho_Sref * Theta_m / gamma # uplift sensitivity factor [k m-3]
        physical_constants['Hw'] = Hw    # vapor scale height
        physical_constants['u'] = u      # x-component of wind vector [m s-1]
        physical_constants['v'] = 0      # y-component of wind vector [m s-1]

        X, Y = np.meshgrid(x, y)
        Orography = h_max * np.exp(-(((X-x0)**2/(2*sigma_x**2))+((Y-y0)**2/(2*sigma_y**2))))
        U = np.multiply(np.ones( (len(Orography), len(Orography[1,:])), dtype = float), physical_constants['u'])
        V = np.multiply(np.ones( (len(Orography), len(Orography[1,:])), dtype = float), physical_constants['v'])

        OP = OrographicPrecipitation(X, Y, U, V, Orography, physical_constants)
        inunit = OP.P_units
        ounit = 'mm hr-1'
        iu = Unit(inunit)
        ou = Unit(ounit)
        P = iu.convert(OP.P, ou)
        
        Pdata.append(P[nx/2+1,:])
        name_str =  '_'.join(['_'.join([k, str(v)]) for k, v in physical_constants.items()])
                     
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        c1 = ax1.imshow(Orography)
        c2 = ax2.imshow(P, cmap=cm.RdBu_r, vmin =-2.5, vmax=2.5)
        ax2.contour(P, np.arange(0.025, 2.025, 0.4), vmin =-2.5, vmax=2.5, colors='k', linestyles='dotted')
        ax1.vlines(132, 0, 400,  colors='0.75', linestyles='dashed')
        ax2.vlines(132, 0, 400,  colors='0.75', linestyles='dashed')
        c3 = ax2.contour(OP.P, [.0001], colors='k')
        ax1.text(.05,0.8, name_str, transform=ax1.transAxes)
        cbar1 = plt.colorbar(c1, ax=ax1)
        cbar2 = plt.colorbar(c2, ax=ax2)
        cbar2.set_label('Precip ({})'.format(ounit))
        outname = name_str + '.pdf'
        fig.savefig(outname)
                     
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in range(len(Pdata)):
        ax.plot(x, Pdata[k])
    
                     
    plt.show()




