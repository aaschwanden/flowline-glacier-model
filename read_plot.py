#
# Flowline glacier model with Linear Orographic Precipitation
#
# Andy Aschwanden, University of Alaska Fairbanks
#
# this code is based on the work of:
# glacier flow model: Doug Brinkerhoff, University of Alaska Fairbanks
# orographic precipitation model: Leif Anderson, University of Iceland
#

from dolfin import *
from argparse import ArgumentParser
import numpy as np
import matplotlib
import matplotlib.animation as animation
import pylab as plt
set_log_level(30)

parser = ArgumentParser()
parser.add_argument('-i', dest='infile',
                    help='File to read in', default=None)

options = parser.parse_args()
infile = options.infile

#
# RESTART    #################################
#

mesh = Mesh()

hdf = HDF5File(mpi_comm_world(), infile, 'r')
hdf.read(mesh, 'mesh', False)
Q = FunctionSpace(mesh, 'CG', 1)
F = Function(Q)

attr = hdf.attributes('H0')
nsteps = attr['count']
Hdata = []
Bdata = []
for i in range(nsteps):
    hdf.read(F, 'B/vector_{}'.format(i))
    Bdata.append(project(F).vector().array())
    hdf.read(F, 'H0/vector_{}'.format(i))
    Hdata.append(project(F).vector().array())

del hdf
    
