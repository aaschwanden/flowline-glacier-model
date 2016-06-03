# ######################################################################
# Flowline glacier model with Linear Orographic Precipitation
#
# Andy Aschwanden, University of Alaska Fairbanks
#
# this code is based on the work of:
# glacier flow model: Doug Brinkerhoff, University of Alaska Fairbanks
# orographic precipitation model: Leif Anderson, University of Iceland
# ######################################################################

from dolfin import *
from argparse import ArgumentParser
import numpy as np
import matplotlib
import matplotlib.animation as animation
from scipy.interpolate import interp1d
import pickle
import pylab as plt
from linear_orog_precip import OrographicPrecipitation
set_log_level(30)
import sys
sys.setrecursionlimit(10000)

parser = ArgumentParser()
parser.add_argument('-i', dest='init_file',
                    help='File with inital state', default=None)
parser.add_argument('-o', dest='out_file',
                    help='Output file', default='out')
parser.add_argument('--smb', dest='precip_model',
                    choices=['linear', 'orog'],
                    help='Precip model', default='linear')
parser.add_argument('--geom', dest='geom',
                    choices=['sym', 'asym', '1sided'],
                    help='Bed geometry.', default='sym')
parser.add_argument('-a', '--t_start', dest='ta', type=float,
                    help='Start year', default=0.)
parser.add_argument('-e', '--t_end', dest='te', type=float,
                    help='End year', default=250.)
parser.add_argument('--dt', dest='dt', type=float,
                    help='Time step', default=1.0)
parser.add_argument('--erosion', dest='erosion', action='store_true',
                    help='Turn on erosion', default=False)

options = parser.parse_args()
init_file = options.init_file
out_file = options.out_file
geom = options.geom
precip_model = options.precip_model
ta = options.ta
te = options.te
dt_float = np.abs(options.dt)  # ensure positivity of time step
erosion = options.erosion

precip_scale_factor = 2  # Tuning factor for magnitude
update_lag = 5

erosion_constants = dict()
erosion_constants['K'] = 2.7e-7
erosion_constants['l'] = 2.


ltop_constants = dict()
ltop_constants['tau_c'] = 500  # conversion time [s]
ltop_constants['tau_f'] = 500  # fallout time [s]
ltop_constants['Nm'] = 0.001       # 0.005 # moist stability frequency [s-1]
ltop_constants['Cw'] = 0.001   # uplift sensitivity factor [k m-3]
ltop_constants['Hw'] = 1000    # vapor scale height
ltop_constants['u'] = -3       # x-component of wind vector [m s-1]
ltop_constants['v'] = 0        # y-component of wind vector [m s-1]
ltop_constants['amin'] = -6.
ltop_constants['amax'] = 10.
ltop_constants['Smin'] = -400
ltop_constants['Smax'] = 2500
ltop_constants['Sela'] = -300
ltop_constants['Pstar'] = 0.1  # background precip
ltop_constants['Pscale'] = 5   # Precip scale factor
ltop_constants['dt'] = 0.2


def function_from_array(x, y, Q, mesh):
    '''
    Returns a function in FunctionSpace Q and mesh interpolated from array y
    '''
    dim = Q.dim()
    N = mesh.geometry().dim()
    mesh_coor = Q.dofmap().tabulate_all_coordinates(mesh).reshape(dim, N)
    mesh_x = mesh_coor[:, 0]
    fx_dofs = Q.dofmap().dofs()
    f_interp = interp1d(x, y)
    mesh_values = f_interp(mesh_x)
    my_f  = Function(Q)
    my_f.vector()[fx_dofs] = mesh_values
    
    return my_f


def array_from_function(f, Q, mesh):
    '''
    Returns a function in FunctionSpace Q and mesh interpolated from array y
    '''
    
    dim = Q.dim()
    N = mesh.geometry().dim()
    mesh_coor = Q.dofmap().tabulate_all_coordinates(mesh).reshape(dim, N)
    mesh_x = mesh_coor[:, 0]
    mesh_y = f.vector().array()
    
    return mesh_x, mesh_y


def get_adot_from_orog_precip(ltop_constants):
    '''
    Calculates SMB for Linear Orographic Precipitation Model
    '''

    amin = ltop_constants['amin']
    amax = ltop_constants['amax']
    Smin = ltop_constants['Smin']
    Smax = ltop_constants['Smax']
    Sela = ltop_constants['Sela']
    Pscale = ltop_constants['Pscale']
    Pstar = ltop_constants['Pstar']

    x_a, y_a = array_from_function(project(S, Q), Q, mesh)
   
    XX, YY = np.meshgrid(x_a, range(3))
    Orography = np.tile(y_a, (3, 1))

    UU = np.multiply(np.ones( (len(Orography), len(Orography[1,:])), dtype = float), ltop_constants['u'])
    VV = np.multiply(np.ones( (len(Orography), len(Orography[1,:])), dtype = float), ltop_constants['v'])    
    OP = OrographicPrecipitation(XX, YY, UU, VV, Orography, ltop_constants)

    P = OP.P
    P = P[1, :] * Pscale + Pstar

    smb_S =  function_from_array(x_a, P, Q, mesh)

    return smb_S, P

##########################################################
###############   Dolfin options       ###################
##########################################################

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['representation'] = 'quadrature'
parameters['allow_extrapolation'] = True

ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


##########################################################
###############        CONSTANTS       ###################
##########################################################

L = 75000.            # Length scale [m]

spy = 31556925.9747   # seconds per year [s year-1]
thklim = 5.0          # Minimum thickness [m]
g = 9.81              # gravity [m s-1]

zmin = -500.0         # SMB parameters
amin =  -8.0           # [m year-1]
amax =  10.0           # [m year-1]
c = 2.0

rho = 900.            # ice density [kg m-3]
rho_w = 1000.0        # water density [kg m-3]

n = 3.0               # Ice material properties
m = 1.0
b = 1e-16**(-1./n)    # ice hardness
eps_reg = 1e-5

dt = Constant(0)                     # Constant time step (gets changed below)

    
#########################################################
#################      GEOMETRY     #####################
#########################################################

my_dx = 1000.  # [m]
x = np.arange(-L, L + my_dx, my_dx)  # [m]
amp = 100.0           # Geometry oscillation parameters
zmax = 2500.  # [m]
x0 = 0
sigma_x = 15e3
sigma_x1 = 5e3
sigma_x2 = 15e3

# Correlation matrix for random topography
N = len(x)
corr_len = 2000.0
corr = np.zeros((N,N))
for i in range(N):
    for j in range(i+1):
        corr[i,j] = exp(-abs(x[i]-x[j])**2/corr_len**2)
        corr[j,i] = exp(-abs(x[i]-x[j])**2/corr_len**2)

# Amplitude of random perturbations
rand_amp = 0.0
cov = rand_amp**2 * corr
z_noise = np.random.multivariate_normal(np.zeros(N),cov)
iii = interp1d(x,z_noise)

# Bed elevation Expression
class BedSym(Expression):
  def eval(self, values, x):
    values[0] = zmax * exp(-(((x[0]-x0)**2/(2*sigma_x**2)))) + zmin
    
class BedAsym(Expression):
  def eval(self, values, x):
    values[0] = zmax * conditional(gt(x[0], 0), exp(-(((x[0]-x0)**2/(2*sigma_x1**2)))), exp(-(((x[0]-x0)**2/(2*sigma_x2**2))))) + zmin

class Bed1Sided(Expression):
  def eval(self,values,x):
    values[0] = (zmax - zmin)*exp(-(x[0]+L)/(L*0.3)) + zmin - amp*(sin(4*pi*x[0]/L)) + iii(x[0])

# Basal traction Expression
class Beta2(Expression):
  def eval(self, values, x):
    values[0] = 5e3

# Flowline width Expression - only relevent for continuity: lateral shear not considered
class Width(Expression):
  def eval(self, values, x):
    values[0] = 1


##########################################################
################           MESH          #################
##########################################################  

# Define a rectangular mesh
nx = 1500                                  # Number of cells
mesh = IntervalMesh(nx, -L, L)            # Equal cell size

X = SpatialCoordinate(mesh)               # Spatial coordinate

ocean = FacetFunctionSizet(mesh, 0)       # Facet function for boundary conditions
ds = ds[ocean] 

# Label the left and right boundary as ocean
for f in facets(mesh):
    if near(f.midpoint().x(), L):
       ocean[f] = 1
       if near(f.midpoint().x(), -L):
           if geom in '1sided':
            ocean[f] = 2
           else:
            ocean[f] = 1
               

# Facet normals
normal = FacetNormal(mesh)

#########################################################
#################  FUNCTION SPACES  #####################
#########################################################

Q = FunctionSpace(mesh, "CG", 1)
V = MixedFunctionSpace([Q]*3)           # ubar,udef,H space

ze = Function(Q)                        # Zero constant function

grounded = Function(Q)                 # Boolean grounded function 
grounded.vector()[:] = 1

if geom in 'sym':
    B = interpolate(BedSym(), Q)         # Bed elevation function
elif geom in 'asym':
    B = interpolate(BedAsym(), Q)
elif geom in '1sided':
    B = interpolate(Bed1Sided(), Q)
else:
    print('{} not supported'.format(geom))
    
beta2 = interpolate(Beta2(), Q)          # Basal traction function

#########################################################
#################  FUNCTIONS  ###########################
#########################################################

# VELOCITY 
U = Function(V)                        # Velocity function
dU = TrialFunction(V)                  # Velocity trial function
Phi = TestFunction(V)                  # Velocity test function

u, u2, H = split(U)                       
phi, phi1, xsi = split(Phi)

un = Function(Q)                       # Temp velocities
u2n = Function(Q)

H0 = Function(Q)
H0.vector()[:] = rho_w/rho * thklim + 1e-3 # Initial thickness

theta = Constant(0.5)                  # Crank-Nicholson
Hmid = theta*H + (1-theta)*H0

# Ice upper surface
S = B + Hmid

# Test and trial functions
psi = TestFunction(Q)                  # Scalar test function
dg = TrialFunction(Q)                  # Scalar trial function

ghat = Function(Q)                     # Temp grounded 
gl = Constant(0)                       # Scalar grounding line

Smax = 2500.    # above Smax, adot=amax [m]
Smin =  200.    # below Smin, adot=amin [m]
Sela = 1000.    # equilibrium line altidue [m]

bmelt = -150.   # sub-shelf melt rate [m year-1]

if precip_model in 'linear':
    adot = conditional(lt(S, Sela), (-amin / (Sela -Smin)) * (S - Sela), (amax / (Smax - Sela)) * (S - Sela)) * grounded +  conditional(lt(S, Sela), (-amin / (Sela -Smin)) * (Hmid - Sela), (amax / (Smax - Sela)) * (Hmid * (1 - rho / rho_w) - Sela)) * (1 - grounded)
    bdot = Constant(0.)
elif precip_model in 'orog':
    adot, P = get_adot_from_orog_precip(ltop_constants)
    bdot = conditional(gt(Hmid, np.abs(bmelt)), bmelt * dt, -Hmid) * (1 - grounded)
else:
    print('precip model {} not supported'.format(precip_model))

 
########################################################
#################   Numerics   #########################
########################################################

# Heuristic spectral element basis
class VerticalBasis(object):
    def __init__(self,u,coef,dcoef):
        self.u = u
        self.coef = coef
        self.dcoef = dcoef

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dx(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])


# Vertical quadrature utility for integrating VerticalBasis class
class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,s,w):
        return w*f(s)
    def intz(self,f):
        return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])


# Surface elevation gradients in z for coordinate change Jacobian
def dsdx(s):
    return 1./Hmid*(S.dx(0) - s*H.dx(0))

def dsdz(s):
    return -1./Hmid

# Ansatz spectral elements (and derivs.): Here using SSA (constant) + SIA ((n+1) order polynomial)
# Note that this choice of element means that the first term is depth-averaged velocity, and the second term is deformational velocity
coef = [lambda s:1.0, lambda s:1./4.*(5*s**4-1.)]
dcoef = [lambda s:0.0, lambda s:5*s**3]

u_ = [U[0],U[1]]
phi_ = [Phi[0],Phi[1]]

# Define function and test function in vertical
u = VerticalBasis(u_,coef,dcoef)
phi = VerticalBasis(phi_,coef,dcoef)

# Quadrature points 
points = np.array([0.0,0.4688,0.8302,1.0])
weights = np.array([0.4876/2.,0.4317,0.2768,0.0476])

vi = VerticalIntegrator(points, weights)
    

########################################################
#################   Momentum Balance    ################
########################################################

# Viscosity (isothermal)
def eta_v(s):
    return b/2.*((u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
                +0.25*((u.ds(s)*dsdz(s))**2) \
                + eps_reg)**((1.-n)/(2*n))

# Membrane stress
def membrane_xx(s):
    return (phi.dx(s,0) + phi.ds(s)*dsdx(s))*Hmid*eta_v(s)*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)))

# Shear stress
def shear_xz(s):
    return dsdz(s)**2*phi.ds(s)*Hmid*eta_v(s)*u.ds(s)

# Driving stress (grounded)
def tau_dx(s):
    return rho*g*Hmid*S.dx(0)*phi(s)

# Driving stress (floating)
def tau_dx_f(s):
    return rho*g*(1-rho/rho_w)*Hmid*Hmid.dx(0)*phi(s)

# Normal vectors
normalx = (B.dx(0))/sqrt((B.dx(0))**2 + 1.0)
normalz = sqrt(1 - normalx**2)

# Overburden
P_0 = rho*g*Hmid
# Water pressure (ocean only, no basal hydro.)
P_w = Max(-rho_w*g*B,1e-16)

# basal shear stress
tau_b = beta2*u(1)/(1.-normalx**2)*grounded

# Momentum balance residual (Blatter-Pattyn/O(1)/LMLa)
R = (- vi.intz(membrane_xx) - vi.intz(shear_xz) - phi(1)*tau_b - vi.intz(tau_dx)*grounded - vi.intz(tau_dx_f)*(1-grounded))*dx

# shelf front boundary condition
F_ocean_x = 1./2.*rho*g*(1-(rho/rho_w))*H**2*Phi[0]*ds(1)

R += F_ocean_x


#############################################################################
##########################  MASS BALANCE  ###################################
#############################################################################

# SUPG parameters
h = CellSize(mesh)
D = h*abs(U[0])/2.

# Width for including convergence/divergence
width = interpolate(Width(),Q)
area = Hmid*width

# Add the SUPG-stabilized continuity equation to residual
R += ((H-H0)/dt*xsi  - xsi.dx(0)*U[0]*Hmid + D*xsi.dx(0)*Hmid.dx(0) - (adot + bdot - un*H0/width*width.dx(0))*xsi)*dx  + U[0]*area*xsi*ds(1)

# Jacobian of coupled momentum-mass system
J = derivative(R, U, dU)


#####################################################################
############################  GL Dynamics  ##########################
#####################################################################

# CN param for updating flotation condition
theta_g = 0.9

# PTC time step (bigger means faster switch from grounded to floating)
dtau = 0.2

# Flotation condition
ghat = conditional(Or(And(ge(rho*g*H,Max(P_w,1e-16)),ge(H,1.5*rho_w/rho*thklim)),ge(B,1e-16)),1,0)

# Flotation update system
R_g = psi*(dg - grounded + dtau*(dg*theta_g + grounded*(1-theta_g) - ghat))*dx
A_g = lhs(R_g)
b_g = rhs(R_g)

#####################################################################
#########################  I/O Functions  ###########################
#####################################################################

# For moving data between vector functions and scalar functions 
assigner_inv = FunctionAssigner([Q,Q,Q], V)
assigner     = FunctionAssigner(V, [Q,Q,Q])


#####################################################################
######################  Variational Solvers  ########################
#####################################################################

# Define variational solver for the momentum problem

# Ice divide dirichlet bc
bc = DirichletBC(V.sub(2),thklim,lambda x,on: near(x[0],-L) and on)

if geom in '1sided':
    mass_problem = NonlinearVariationalProblem(R,U,bcs=[bc],J=J,
                                               form_compiler_parameters=ffc_options)
else:
    # No Dirichlet BCs for symmetric geometry, both sides are ocean
    mass_problem = NonlinearVariationalProblem(R, U, J=J,
                                           form_compiler_parameters=ffc_options)

# Account for thickness positivity by using vi-newton-rsls solver from PETSc
mass_solver = NonlinearVariationalSolver(mass_problem)
mass_solver.parameters['nonlinear_solver'] = 'snes'

mass_solver.parameters['snes_solver']['method'] = 'vinewtonrsls'
mass_solver.parameters['snes_solver']['relative_tolerance'] = 1e-3
mass_solver.parameters['snes_solver']['absolute_tolerance'] = 1e-3
mass_solver.parameters['snes_solver']['error_on_nonconvergence'] = True
mass_solver.parameters['snes_solver']['linear_solver'] = 'mumps'
mass_solver.parameters['snes_solver']['maximum_iterations'] = 100
mass_solver.parameters['snes_solver']['report'] = False

# Bounds
l_thick_bound = project(Constant(thklim),Q)
u_thick_bound = project(Constant(1e4),Q)

l_v_bound = project(-10000.0,Q)
u_v_bound = project(10000.0,Q)

l_bound = Function(V)
u_bound = Function(V)

assigner.assign(l_bound,[l_v_bound]*2+[l_thick_bound])
assigner.assign(u_bound,[u_v_bound]*2+[u_thick_bound])

x = mesh.coordinates().ravel()
SS = project(S)
us = project(u(0))
ub = project(u(1))

adot_p = project(adot, Q).vector().array()

mass = []
time = []

tdata = []
Hdata = []
Sudata = []
Sldata = []
Bdata = []
ubardata = []
udefdata = []
usdata = []
ubdata = []
gldata = []
grdata = []
adotdata = []
bdotdata = []
Pdata = []


######################################################################
#######################   SOLUTION   #################################
######################################################################

# Time interval
t = ta
t_end = te
dt.assign(dt_float)

assigner.assign(U, [ze,ze,H0])

######################################################################
#######################   RESTART    #################################
######################################################################

if init_file is not None:
    '''
    Restart from file
    '''
    hdf = HDF5File(mpi_comm_world(), init_file, 'r')
    hdf.read(mesh, 'mesh', False)
    hdf.read(H0, 'H0')
    hdf.read(un, 'ubar')
    hdf.read(u2n, 'udef')
    hdf.read(grounded, 'grounded')
    assigner.assign(U,[un,u2n,H0])

# Save the time series
hdf = HDF5File(mesh.mpi_comm(), out_file + '.h5', 'w')
hdf.write(mesh, 'mesh')
    
# Loop over time
i = 0
while t < t_end:
    time.append(t)

    # Update grounding line position
    solve(A_g == b_g, grounded)
    grounded.vector()[0] = 1

    # Hard bed erosion
    if erosion:
        # Only update every update_lag years because
        # this is computationally expensive
        if (np.mod(t, update_lag) == 0):
            K = erosion_constants['K']
            l = erosion_constants['l']
            mdot =  K * abs(ub)**l  * grounded
            B -= mdot * (update_lag/dt)
            print('Erosion rate {} mm year-1'.format(project(mdot).vector().max()*1e3))


    # Try solving with last solution as initial guess for next solution
    try:
        mass_solver.solve(l_bound, u_bound)
    # If this breaks, set initial guess to zero and try again
    except:
        assigner.assign(U,[ze, ze, H0])
        mass_solver.solve(l_bound, u_bound)

    # Set previous time step variables
    assigner_inv.assign([un,u2n,H0],U)

    # Upper glacier surface
    S_u = (B + H) * grounded + H0 * (1 - rho / rho_w) * (1 - grounded)
    # Lower glacier surface
    S_l = B * grounded + H0 * (-rho / rho_w) * (1 - grounded)
    
    us = project(u(0))
    ub = project(u(1))

    
    P = None
    if precip_model in 'orog':
        adot, P = get_adot_from_orog_precip(ltop_constants)
    adot_p = project(adot, Q).vector().array()
    bdot_p = project(bdot, Q).vector().array()
        
    # Save values at each time step
    tdata.append(t)
    Hdata.append(H0.vector().array())
    Sudata.append(project(S_u).vector().array())
    Sldata.append(project(S_l).vector().array())
    Bdata.append(project(B).vector().array())
    gldata.append(gl(0))
    ubardata.append(un.vector().array())
    udefdata.append(u2n.vector().array())
    usdata.append(us.vector().array())
    ubdata.append(ub.vector().array())
    grdata.append(grounded.vector().array())
    adotdata.append(adot_p)
    bdotdata.append(bdot_p)
    Pdata.append(P)

    hdf.write(project(S), 'S', i)
    hdf.write(project(S_l), 'Sl', i)
    hdf.write(project(S_u), 'Su', i)
    hdf.write(project(B), 'B', i)
    hdf.write(H0, 'H0', i)
    hdf.write(un, 'ubar', i)
    hdf.write(u2n, 'udef', i)
    hdf.write(grounded, 'grounded', i)
    i += 1
    
    print('Year {:2.2f}, Hmax {:2.0f}, adotmax {:2.2f}'.format(t, H0.vector().max(), adot_p.max()))
    t += dt_float

del hdf

# # Save relevant data to pickle
# pickle.dump((tdata,Hdata,Sudata,Sldata,Bdata,ubardata,udefdata,usdata,ubdata,grdata,gldata,adotdata,bdotdata), open(out_file + '.p', 'w'))

# Save last time step for restarting purposes
hdf = HDF5File(mesh.mpi_comm(), 'init_' + out_file + '.h5', 'w')
hdf.write(mesh, 'mesh')
hdf.write(project(S), 'S')
hdf.write(project(S_l), 'Sl')
hdf.write(project(S_u), 'Su')
hdf.write(project(B), 'B')
hdf.write(H0, 'H0')
hdf.write(un, 'ubar')
hdf.write(u2n, 'udef')
hdf.write(grounded, 'grounded')

del hdf


# Visualization

def animate(i):
    line_su.set_ydata(Sudata[i])  
    line_sl.set_ydata(Sldata[i])
    line_ub.set_ydata(ubdata[i])
    line_us.set_ydata(usdata[i])
    line_adot.set_ydata(adotdata[i])
    txt.set_text('Year {}'.format(tdata[i]))
    return line_su, line_sl, line_ub, line_us, line_adot, txt

# Init only required for blitting to give a clean slate.
def init():
    line_su.set_ydata(np.ma.array(x_km, mask=True))
    line_sl.set_ydata(np.ma.array(x_km, mask=True))
    line_ub.set_ydata(np.ma.array(x_km, mask=True))
    line_us.set_ydata(np.ma.array(x_km, mask=True))
    line_adot.set_ydata(np.ma.array(x_km, mask=True))
    return line_su, line_sl, line_ub, line_us, line_adot

x_km = x / 1000.
fig, ax = plt.subplots(nrows=3, sharex=True)
ax[0].set_ylim(zmin, 3000)
ax[0].plot(x_km, Bdata[0], 'r')
ax[0].set_ylabel('altitude (m)')
txt = ax[0].text(0.025, 0.75, 'Year         ',
                 transform=ax[0].transAxes)
line_su, = ax[0].plot(x_km, Sudata[0], 'b')
line_sl, = ax[0].plot(x_km, Sldata[0], 'g')
line_ub, = ax[1].plot(x_km, ubdata[0], 'k')
line_us, = ax[1].plot(x_km, usdata[0], 'b')
ax[1].set_ylabel('us, ub (m year-1)')
ax[1].set_ylim(-750, 750)
line_adot, = ax[2].plot(x_km, adotdata[0])
ax[2].set_ylim(-8, 12)
ax[2].set_ylabel('adot (m year-1)')
ax[2].set_xlabel('x (km)')
ani = animation.FuncAnimation(fig, animate,
                              frames=len(tdata),
                              init_func=init,
                              interval=5, blit=True)
ani.save(out_file + '.mp4', fps=24, extra_args=['-vcodec', 'libx264'])
plt.show()

