from D2linear import D2linear
import numpy as np

dx = 750
dy = 750
x = np.r_[-100e3:200e3:dx]
y = np.r_[-150e3:150e3:dy]

X,Y = np.meshgrid(x,y)
sigma_x = 15e3
sigma_y = 15e3
x0=-25e3
y0=0
h_max = 500
u0 = 15
v0 = 0
tau = 2000
T0 = 0
Nm = 0.005

Orography = h_max * np.exp(-(((X-x0)**2/(2*sigma_x**2))+((Y-y0)**2/(2*sigma_y**2))))

precip = D2linear(Orography, u0, v0, tau, T0, Nm)

import pylab as plt

fig = plt.figure()
ax = fig.add_subplot(111)
c = ax.imshow(precip)
plt.colorbar(c)
ax.contour(Orography, np.arange(0, 600, 100), colors='0.')
cs = ax.contour(precip, np.arange(0.025, 2.425, 0.4), colors='1.')
plt.clabel(cs, inline=1, fontsize=10)
plt.show()
