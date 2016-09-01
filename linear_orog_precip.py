import logging
import logging.handlers
from scipy.fftpack import fft2, fftfreq
import numpy as np
import cmath
from osgeo import gdal, osr

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.handlers.RotatingFileHandler('ltop.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')

# add formatter to ch and fh
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)
logger.addHandler(fh)

class OrographicPrecipitation(object):

    """Calculates orographic precipitation following Smith & Barstad (2004).

    """

    def __init__(self, X, Y, Orography, physical_constants, truncate=True, ounits=None):
        self.logger = logger or logging.getLogger(__name__, logger=None)
        self.logger.info('Initializing instance of OrographicPrecipitation')
        self.X = X
        self.Y = Y
        self.Orography = Orography
        self.physical_constants = physical_constants
        self.dx = np.diff(X)[0, 0]
        self.dy = np.diff(Y, axis=0)[0, 0]
        self.nx = len(Orography[1, :])
        self.ny = len(Orography)
        self.truncate = truncate

        self.P = self._compute_precip(ounits)
        if ounits is not None:
            self.P_units = ounits
        else:
            self.P_units = 'mm hr-1'

    def _regularize(self, x):
        eps = 1e-18
        if np.fabs(x) < 1e-18:
            if np.fabs(x) >= 0:
                return eps
            else:
                return -eps
        else:
            return x

    def _compute_precip(self, ounits):

        physical_constants = self.physical_constants
        Orography = self.Orography
        logger.info('Fourier transform orography')
        Orography_fft = np.fft.fft2(Orography)
        dx = self.dx
        dy = self.dy
        nx = self.nx
        ny = self.ny

        # pre-compute kx
        dkx = 2.0 * np.pi / (nx * dx)
        kx = np.zeros(nx)
        nchx = (nx / 2) + 1
        for i in xrange(0, nchx):
            kx[i] = i * dkx
        for i in xrange(nchx, nx):
            kx[i] = - (nx - i) * dkx

        # pre-compute ky
        dky = 2.0 * np.pi / (ny * dy)
        ky = np.zeros(ny)
        nchy = (ny / 2) + 1
        for i in xrange(0, nchy):
            ky[i] = i * dky
        for i in xrange(nchy, ny):
            ky[i] = - (ny - i) * dky

        h_hat = fft2(Orography)
        P_hat = np.zeros_like(h_hat)

        for j in xrange(ny):
            for i in xrange(nx):

                sigma = physical_constants['u'] * kx[i] + physical_constants['v'] * ky[j]

                C = self._regularize(sigma**2.0)

                m_squared = ((Nm**2 - sigma**2) / C) * (kx[i]**2 + ky[j]**2)

                if m_squared >= 0.0:
                    if sigma == 0.0:
                        m = np.sqrt(m_squared)
                    else:
                        m = np.sign(sigma) * np.sqrt(m_squared)
                else:
                    m = np.sqrt(-1 * m_squared)

                assert np.abs(1 - 1j * m * Hw) > 1e-10

                # see equation 49
                D = (1.0 - 1j * m * physical_constants['Hw']) * (1.0 + 1j * sigma * physical_constants['tau_c']) * (1.0 + 1j * sigma * physical_constants['tau_f'])
                P_hat[j, i] = physical_constants['Cw'] * 1j * sigma * h_hat[j, i] / D

        P = np.real(np.fft.ifft2(P_hat)) * 3600.0
        P = np.maximum(P, 0.0)
        # # Add background precip
        # P += physical_constants['P0']
        # # Truncation
        # truncate = self.truncate
        # if truncate is True:
        #     P[P < 0] = 0
        # P_scale = physical_constants['P_scale']
        # P *= P_scale 

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
        logger.info('Initializing instance of GdalFile')
        self.file_name = file_name
        try:
            print("\n  opening file %s" % file_name)
            self.ds = gdal.Open(file_name)
        except:
            print("  could not open file %s" % file_name)

        self.RasterArray = self.ds.ReadAsArray()
        self.projection = self.ds.GetProjection()
        logger.info('Found projection {}'.format(self.projection))

        geoT = self.ds.GetGeoTransform()
        logger.info('GeoTransform is {}'.format(geoT))
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

    logger.info('Saving {} as netcdf file'.format(newRasterfn))
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
    logger.info('Write {} to disk'.format(newRasterfn))
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
    parser.add_argument('--background_precip', dest='P0', type=float,
                        help='Background precipitation rate [mm hr-1].', default=0.)
    parser.add_argument('--precip_scale_factor', dest='P_scale', type=float,
                        help='Precipitation scale factor.', default=1.)
    parser.add_argument('--no_trunc', dest='truncate', action='store_false',
                        help='Do not truncate precipitation.', default=True)
    parser.add_argument('--latitude', dest='lat', type=float,
                        help='Latitude to compute Coriolis term.', default=45.)
    parser.add_argument('--tau_c', dest='tau_c', type=float,
                        help='conversion time [s].', default=1000)
    parser.add_argument('--tau_f', dest='tau_f', type=float,
                        help='fallout time [s].', default=1000)
    parser.add_argument('--moist_stability', dest='Nm', type=float,
                        help='moist stability frequency [s-1].', default=0.005)
    parser.add_argument('--vapor_scale_height', dest='Hw', type=float,
                        help='Water vapor scale height [m].', default=2500)
    parser.add_argument('--wind_direction', dest='direction', type=float,
                        help='Direction from which the wind is coming.', default=270)
    parser.add_argument('--wind_magnitude', dest='magnitude', type=float,
                        help='Magnitude of wind velocity [m/s].', default=15)


    logger.info('Parsing options.')
    options = parser.parse_args()
    in_file = options.in_file
    out_file = options.out_file
    direction = options.direction
    lat = options.lat
    magnitude = options.magnitude
    tau_c = options.tau_c
    tau_f = options.tau_f
    truncate = options.truncate
    Nm = options.Nm
    Hw = options.Hw
    P0 = options.P0
    P_scale = options.P_scale
    logger.info('Parsing options. Done.')

    
    if in_file is not None:
        gd = GdalFile(in_file)
        X = gd.X
        Y = gd.Y
        Orography = gd.RasterArray
    else:
        # Reproduce Fig 4c in SB2004
        dx = dy = 750.
        x, y = np.arange(-100e3, 200e3, dx), np.arange(-150e3, 150e3, dy)
        h_max = 500.
        x0 = -25e3
        y0 = 0
        sigma_x = sigma_y = 15e3
        X, Y = np.meshgrid(x, y)
        Orography = h_max * np.exp(-(((X-x0)**2/(2*sigma_x**2))+((Y-y0)**2/(2*sigma_y**2))))

    Theta_m = -6.5     # K / km
    rho_Sref = 7.4e-3  # kg m-3
    gamma = -5.8       # K / km

    physical_constants = dict()
    physical_constants['tau_c'] = tau_c      # conversion time [s]
    physical_constants['tau_f'] = tau_f      # fallout time [s]
    physical_constants['f'] = 2 * 7.2921e-5 * np.sin(lat * np.pi / 180) # Coriolis force
    physical_constants['Nm'] = Nm   # moist stability frequency [s-1]
    physical_constants['Cw'] = rho_Sref * Theta_m / gamma # uplift sensitivity factor [kg m-3]
    physical_constants['Hw'] = Hw         # vapor scale height
    physical_constants['u'] = -np.sin(direction*2*np.pi/360) * magnitude    # x-component of wind vector [m s-1]
    physical_constants['v'] = -np.cos(direction*2*np.pi/360) * magnitude   # y-component of wind vector [m s-1]
    physical_constants['P0'] = P0   # background precip [mm hr-1]
    physical_constants['P_scale'] = P_scale   # precip scale factor [1]
    logger.debug('Physical constants: {}'.format(physical_constants))
    
    OP = OrographicPrecipitation(X, Y, Orography, physical_constants, truncate=truncate)
    P = OP.P
    units = OP.P_units

    if in_file is not None:
        array2raster(out_file, gd.geoTrans, gd.proj4, units, P)
    else:
        geoTrans = [0., OP.dx, 0., 0., 0., -OP.dy]
        array2raster(out_file, geoTrans, '', units, P)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        c = ax.pcolormesh(X, Y, P)
        plt.colorbar(c)
        ax.contour(X, Y, Orography, np.arange(0, 600, 100), colors='0.')
        cs = ax.contour(X, Y, P, np.arange(0.025, 2.425, 0.4), colors='1.')
        plt.clabel(cs, inline=1, fontsize=10)
        ax.set_xlim(-100e3, 200e3)
        ax.set_ylim(-150e3, 150e3)
        plt.show()
