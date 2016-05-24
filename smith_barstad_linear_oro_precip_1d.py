# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:47:27 2015

@author: leif
"""

#clear_all()

from scipy.fftpack import fft2, fftfreq
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cmath

dx = dy = 1000.
L = 50.e3
x, y = np.arange(-L, L, dx), np.arange(-L, L, dx)
nx, nx = len(x), len(y)
h_max = L / 50
x0 = y0 = 0
sigma_x = sigma_y = L / 4


physical_constants = dict()
physical_constants['tau_c'] = 1000  # [s]
physical_constants['tau_f'] = 2000  # [s]
physical_constants['f'] = 2 * 7.2921e-5 * np.sin(60 * np.pi / 180)
physical_constants['Nm'] = 0  # 0.005 # moist stability frequency [s-1]
physical_constants['Cw'] = 0.001  # uplift sensitivity factor [k m-3]
physical_constants['Hw'] = 1000  # vapor scale height
physical_constants['u'] = 5  # x-component of wind vector [m s-1]
physical_constants['v'] = 0  # y-component of wind vector [m s-1]




X, Y = np.meshgrid(x, y)


H = h_max * np.exp(-(((X-x0)**2/(2*sigma_x**2))+((Y-y0)**2/(2*sigma_y**2))))
myarray = H


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# surf = ax.plot_surface(X, Y, H, rstride=1, cstride=1, cmap=cm.coolwarm,
#         linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

plt.figure()
plt.imshow(myarray)
cbar=plt.colorbar()
cbar.set_label('Elevation (m)', rotation=270, labelpad=20)
plt.xlabel('Distance (m)')
plt.ylabel('Distanve (m)')


ny = len(myarray)
nx = len(myarray[1,:])


myarray = H

myarrayfft = np.fft.fft2(myarray)

x_n_value = np.fft.fftfreq(len(myarray[1,:]), (1.0 / len(myarray[1,:])))
y_n_value = np.fft.fftfreq(len(myarray), (1.0/len(myarray)))

# x_array = np.zeros((len(myarray), len(myarray[1,:])), dtype = float)
# y_array = np.zeros((len(myarray), len(myarray[1,:])), dtype = float)

U = np.multiply(np.ones( (len(myarray), len(myarray[1,:])), dtype = float), physical_constants['u'])
V = np.multiply(np.ones( (len(myarray), len(myarray[1,:])), dtype = float), physical_constants['v'])

x_len = len(myarray) * dx
y_len = len(myarray[1,:]) * dy

kx_line = np.divide(np.multiply(2.0 * np.pi, x_n_value), x_len)
ky_line = np.divide(np.multiply(2.0 * np.pi, y_n_value), y_len)[np.newaxis].T

kx = np.tile(kx_line, (ny, 1))
ky = np.tile(ky_line, (1, nx))

sigma = np.add(np.multiply(kx, U), np.multiply(ky, V))

m = np.power(np.multiply(np.add(np.power(kx, 2.) , np.power(ky, 2.)) , np.divide(np.subtract(physical_constants['Nm']**2., np.power(sigma, 2.)) , np.subtract(np.power(sigma, 2.),physical_constants['f']**2.))) , 0.5)

m[np.isnan(m)] = 0
m = np.maximum(m,0.0001)
m = np.minimum(m,0.01)


P_karot_nom = np.multiply(np.multiply(np.multiply(physical_constants['Cw'], cmath.sqrt(-1)), sigma), myarrayfft)
P_karot_denom_Hw = np.subtract(np.multiply(np.multiply(physical_constants['Hw'], m), cmath.sqrt(-1)), 1)
P_karot_denom_tauc =  np.add(np.multiply(np.multiply(sigma, physical_constants['tau_c']), cmath.sqrt(-1)), 1)
P_karot_denom_tauf = np.add(np.multiply(np.multiply(sigma, physical_constants['tau_f']), cmath.sqrt(-1)), 1)
P_karot_denom = np.multiply(P_karot_denom_Hw, np.multiply(P_karot_denom_tauc, P_karot_denom_tauf))
P_karot = np.divide(P_karot_nom, P_karot_denom)      

P_karot_amp = np.absolute(P_karot) # get the amplitude
P_karot_angle = np.angle(P_karot) # get the phase angle

y2 = np.multiply(P_karot_amp,  np.add(np.cos(P_karot_angle) , np.multiply(cmath.sqrt(-1) , np.sin(P_karot_angle))))

y3 = np.fft.ifft2(y2)

#y3 = np.fft.ifft(P_karot)

P = np.multiply(np.real(y3),(60*60*24*100)/1000)   #(np.pi*10**7)/1000/12)  # times 100 for cm
P[P<0] = 0
P[myarray<1] = 0

#plt.imshow(myarrayfft)
plt.figure(2)
#imshow(P[300:400,350:450])
plt.imshow(P)
cbar = plt.colorbar()
cbar.set_label('Precip (cm/day)', rotation=270, labelpad=20)

plt.xlabel('Distance (km)')
plt.ylabel('Distanve (km)')
