# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 18:47:27 2015

@author: leif
"""


from scipy.fftpack import fft2, fftfreq
import numpy as np
import cmath
from osgeo import gdal, osr


class OrographicPrecipitation(object):

    """Calculates orographic precipitation following Smith & Barstad (2004).

    """

    def __init__(self, X, Y, U, V, Orography, physical_constants, ounits=None):
        self.X = X
        self.Y = Y
        self.U = U
        self.V = V
        self.Orography = Orography
        self.physical_constants = physical_constants
        self.dx = np.diff(X)[0, 0]
        self.dy = np.diff(Y, axis=0)[0, 0]
        self.nx = len(Orography[1,:])
        self.ny = len(Orography)

        self.P = self._compute_precip(ounits)
        if ounits is not None:
            self.P_units = ounits
        else:
            self.P_units = 'mm hr-1'
        
    def _compute_precip(self, ounits):

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
        m = np.power(np.multiply(np.divide(np.subtract(physical_constants['Nm']**2, np.power(sigma, 2.)), np.power(sigma, 2.)), np.add(np.power(kx, 2.), np.power(ky, 2.))), 0.5)
        
        m[np.isnan(m)] = 0
        # m = np.maximum(m,0.00001)
        m = np.minimum(m, 0.01)
        
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
        y2 = np.multiply(P_karot_amp,  np.add(np.cos(P_karot_angle), np.multiply(cmath.sqrt(-1), np.sin(P_karot_angle))))
        y3 = np.fft.ifft2(y2)
        spy = 31556925.9747
        P = np.multiply(np.real(y3), 3600)   # mm hr-1
        # Truncation
        P[P < 0] = 0
        # Add background precip
        P +=  physical_constants['P0']
        if physical_constants['Pscale']:
            P *= physical_constants['Pscale']

        if ounits is not None:
            import cf_units
            in_units = cf_units.Unit('mm hr-1')
            out_units = cf_units.Unit(ounits)
            P = in_units.convert(P, out_units)
        return P


class GdalFile(object):

    '''
    A class to read a GDAL File

    Parameters
    ----------

    filename: a valid gdal file
    '''

    def __init__(self, file_name):

        self.file_name = file_name
        try:
            print("\n  opening file %s" % file_name)
            self.ds = gdal.Open(file_name)
        except:
            print("  could not open file %s" % file_name)

        self.RasterArray = self.ds.ReadAsArray()
        self.projection = self.ds.GetProjection()

        geoT = self.ds.GetGeoTransform()
        pxwidth = self.ds.RasterXSize
        pxheight = self.ds.RasterYSize
        ulx = geoT[0]
        uly = geoT[3]
        rezX = geoT[1]
        rezY = geoT[5]
        rx = ulx + pxwidth * rezX
        ly = uly + pxheight * rezY
        osr_ref = osr.SpatialReference()
        osr_ref.ImportFromWkt(self.projection)
        self.proj4 = osr_ref.ExportToProj4()

        self.geoTrans = geoT
        self.width = np.abs(pxwidth * rezX)
        self.height = np.abs(pxheight * rezY)
        self.center_x = ulx + pxwidth * rezX / 2
        self.center_y = uly + pxheight * rezY / 2
        self.easting = np.arange(ulx, rx + rezX, rezX)
        self.northing = np.arange(ly, uly - rezY, -rezY)
        self.X, self.Y = np.meshgrid(self.easting, self.northing)


def array2raster(newRasterfn, geoTrans, proj4, units, array):
    '''
    Function to export geo-coded raster

    Parameters
    ----------

    '''

    cols = array.shape[1]
    rows = array.shape[0]

    driver = gdal.GetDriverByName('netCDF')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Float32)
    outRaster.SetGeoTransform((geoTrans))
    outband = outRaster.GetRasterBand(1)
    outband.SetMetadata('units', units)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromProj4(proj4)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
        
if __name__ == "__main__":
    print('Linear Orographic Precipitation Model by Smith & Barstad (2004)')

    import pylab as plt
    from matplotlib import cm
    import itertools
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-i', dest='in_file',
                        help='Gdal-readable DEM', default=None)
    parser.add_argument('-o', dest='out_file',
                        help='Output file', default='foo.nc')
    parser.add_argument('--wind_direction', dest='direction', type=float,
                        help='Direction from which the wind is coming.', default=225.)
    parser.add_argument('--wind_magnitude', dest='magnitude', type=float,
                        help='Magnitude of wind velocity [m/s].', default=15.)
    parser.add_argument('--tau_c', dest='tau_c', type=float,
                        help='conversion time [s].', default=1000.)
    parser.add_argument('--tau_f', dest='tau_f', type=float,
                        help='fallout time [s].', default=1000.)
    parser.add_argument('--moist_stability', dest='Nm', type=float,
                        help='moist stability frequency [s-1].', default=0.0032)
    parser.add_argument('--vapor_scale_height', dest='Hw', type=float,
                        help='Water vapor scale height [m].', default=2500)
    parser.add_argument('--background_precip', dest='P0', type=float,
                        help='Background precipitation rate [m/s].', default=0.)
    
    options = parser.parse_args()
    in_file = options.in_file
    out_file = options.out_file
    direction = options.direction
    magnitude = options.magnitude
    tau_c = options.tau_c
    tau_f = options.tau_f
    Nm = options.Nm
    Hw = options.Hw
    P0 = options.P0
    
    if in_file is not None:
        gd = GdalFile(in_file)
        X = gd.X
        Y = gd.Y
        Orography = gd.RasterArray
    else:
        dx = dy = 750.
        x, y = np.arange(-100e3, 200e3, dx), np.arange(-150e3, 150e3, dy)
        nx, nx = len(x), len(y)
        h_max = 500.
        x0 = y0 = 0
        sigma_x = sigma_y = 15e3
        X, Y = np.meshgrid(x, y)
        Orography = h_max * np.exp(-(((X-x0)**2/(2*sigma_x**2))+((Y-y0)**2/(2*sigma_y**2))))

    Theta_m = -6.5     # K / km
    rho_Sref = 7.4e-3  # kg m-3
    gamma = -5.8       # K / km

    
    physical_constants = dict()
    physical_constants['tau_c'] = tau_c      # conversion time [s]
    physical_constants['tau_f'] = tau_f      # fallout time [s]
    physical_constants['Nm'] = Nm       # moist stability frequency [s-1]
    physical_constants['Cw'] = rho_Sref * Theta_m / gamma # uplift sensitivity factor [kg m-3]
    physical_constants['Hw'] = Hw         # vapor scale height
    physical_constants['u'] = np.sin(direction*2*np.pi/360) * magnitude    # x-component of wind vector [m s-1]
    physical_constants['v'] = -np.cos(direction*2*np.pi/360) * magnitude   # y-component of wind vector [m s-1]
    physical_constants['P0'] = P0   # background precip [m s-1]

    U = np.multiply(np.ones( (len(Orography), len(Orography[1,:])), dtype = float), physical_constants['u'])
    V = np.multiply(np.ones( (len(Orography), len(Orography[1,:])), dtype = float), physical_constants['v'])

    OP = OrographicPrecipitation(X, Y, U, V, Orography, physical_constants)
    P = OP.P
    units = OP.P_units

    array2raster(out_file, gd.geoTrans, gd.proj4, units, P)



