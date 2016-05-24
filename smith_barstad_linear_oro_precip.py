# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:47:27 2015

@author: leif
"""

#clear_all()

from scipy.fftpack import fft2, fftfreq
import numpy as np
import pylab as plt
from osgeo import gdal
import cmath
#ds = gdal.Open("srtm_mosaik_utm.tif")
#myarray = np.array(ds.GetRasterBand(1).ReadAsArray())

#grid_x, grid_y = np.mgrid[0:1:9890j, 0:1:12819j]


grid_xi, grid_yi = np.mgrid[10.:10.:9890., 10.:10.:12819.]


#mygridded = scipy.interpolate.griddata(values,myarray,(grid_xi,grid_yi),method='linear')

ds = gdal.Open("srtm_mosaik_isn93_1km_dem_best.tif")
myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
myarray[myarray>5000]=0
#myarray[myarray<1]=1

dx = 1000
dy = 1000

#import matplotlib.pyplot as plt

plt.figure()
plt.imshow(myarray)
cbar=plt.colorbar()
cbar.set_label('Elevation (m)', rotation=270, labelpad=20)
plt.xlabel('Distance (km)')
plt.ylabel('Distanve (km)')

#some_data_wave_domain = fft2(myarray)


#np.tile(x,(1,num_y))

num_y = len(myarray)
num_x = len(myarray[1,:])


myarrayfft = np.fft.fft2(myarray)

y_n_value = np.fft.fftfreq(len(myarray),(1.0/len(myarray)))
x_n_value = np.fft.fftfreq( len(myarray[1,:]),(1.0/ len(myarray[1,:])))

x_array = np.zeros( (len(myarray), len(myarray[1,:])), dtype = float)
y_array = np.zeros( (len(myarray), len(myarray[1,:])), dtype = float)

V_array = np.multiply(np.ones( (len(myarray), len(myarray[1,:])), dtype = float),5)
U_array = np.multiply(np.ones( (len(myarray), len(myarray[1,:])), dtype = float),0)

#edge = 0
#V_array[edge:num_y-edge, edge:num_x-edge ] = np.add(V_array[edge:num_y-edge, edge:num_x-edge ],0)
#U_array[edge:num_y-edge, edge:num_x-edge ] = np.add(V_array[edge:num_y-edge, edge:num_x-edge],0)

x_len = len(myarray)*dx
y_len = len(myarray[1,:])*dy

kx_line = np.divide(np.multiply(2.0*np.pi,x_n_value),x_len)
ky_line = np.divide(np.multiply(2.0*np.pi,y_n_value),y_len)[np.newaxis].T

kx = np.tile(kx_line,(num_y,1))

ky = np.tile(ky_line,(1,num_x))

sigma = np.add(np.multiply(kx,U_array), np.multiply(ky,V_array))


tau_c = 1000; # in seconds
tau_f = 2000; # in seconds


f = 2* 7.2921e-5*np.sin(60*np.pi/180)
Nm = 0 #0.005 # moist stability frequency in 1/s

Cw = 0.001 # uplift sensitivity factor k/m3

Hw = 1000 # vapor scale height

m = np.power(np.multiply(np.add(np.power(kx,2.) , np.power(ky,2.)) , np.divide(np.subtract(Nm**2., np.power(sigma,2.)) , np.subtract(np.power(sigma,2.),f**2.))) , 0.5)

m[np.isnan(m)] = 0
m = np.maximum(m,0.0001)
m = np.minimum(m,0.01)


P_karot = np.divide(np.multiply(np.multiply(np.multiply(Cw,cmath.sqrt(-1)),sigma),myarrayfft), np.multiply(np.subtract(np.multiply (np.multiply(Hw,m) , cmath.sqrt(-1)), 1),  np.multiply(np.add(np.multiply( np.multiply(sigma,tau_c), cmath.sqrt(-1)),1),np.add(np.multiply( np.multiply(sigma,tau_f), cmath.sqrt(-1)),1))))      

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
