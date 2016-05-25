# Flowline glacier model code by D. Brinkerhoff, UAF

from dolfin import *
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import trapz, cumtrapz
from scipy.interpolate import interp1d
from pylab import argmax, where
import pickle
from linear_orog_precip import OrographicPrecipitation


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

L = 50000.            # Length scale [m]

spy = 60**2*24*365    # seconds per year [s year-1]
thklim = 5.0          # Minimum thickness [m]
g = 9.81              # gravity [m s-1]

zmin = -300.0         # SMB parameters
amin = -8.0           # [m year-1]
amax = 12.0           # [m year-1]
c = 2.0

shore = 0

rho = 900.            # ice density [kg m-3]
rho_w = 1000.0        # water density [kg m-3]

n = 3.0               # Ice material properties
m = 1.0
b = 1e-16**(-1./n)    # ice hardness
eps_reg = 1e-5

#########################################################
#################      GEOMETRY     #####################
#########################################################

dx = 1000.  # [m]
x = np.arange(-L, L + dx, dx)  # [m]
h_max = 1000.  # [m]
x0 = 0
sigma_x = L / 4


# Bed elevation Expression
class Bed(Expression):
  def eval(self, values, x):
    values[0] = h_max * exp(-(((x[0]-x0)**2/(2*sigma_x**2)))) + zmin

# Basal traction Expression
class Beta2(Expression):
  def eval(self,values,x):
    values[0] = 2e3

# Flowline width Expression - only relevent for continuity: lateral shear not considered
class Width(Expression):
  k = 1.0
  theta = 5000.0
  w_min = 3000.0
  xmin = -L
  A = 200e6
  def eval(self,values,x):
    values[0] = 1

##########################################################
################           MESH          #################
##########################################################  

# Define a rectangular mesh
nx = 300                                  # Number of cells
mesh = IntervalMesh(nx, -L, L)            # Equal cell size

X = SpatialCoordinate(mesh)               # Spatial coordinate

ocean = FacetFunctionSizet(mesh, 0)       # Facet function for boundary conditions
ds = ds[ocean] 

# Label the left and right boundary as ocean
for f in facets(mesh):
    if near(f.midpoint().x(), L):
       ocean[f] = 1
    if near(f.midpoint().x(), -L):
       ocean[f] = 1

# Facet normals
normal = FacetNormal(mesh)

#########################################################
#################  FUNCTION SPACES  #####################
#########################################################

Q = FunctionSpace(mesh,"CG", 1)
V = MixedFunctionSpace([Q]*3)           # ubar,udef,H space

ze = Function(Q)                        # Zero constant function

B = interpolate(Bed(),Q)                # Bed elevation function
beta2 = interpolate(Beta2(),Q)          # Basal traction function

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

S = B + Hmid

psi = TestFunction(Q)                  # Scalar test function
dg = TrialFunction(Q)                  # Scalar trial function

grounded = Function(Q)                 # Boolean grounded function 
grounded.vector()[:] = 1

ghat = Function(Q)                     # Temp grounded 
gl = Constant(0)                       # Scalar grounding line

dt = Constant(0.1)                     # Constant time step (gets changed below)

# Surface mass balance expression: If no feedback between elevation and SMB is desired,
# consider interpolating an Expression instead.

# adot = (amin + (amax-amin)/(1-exp(-c))*(1.-exp(-c*((S-0)/(2000.-0)))))*grounded + (amin + (amax-amin)/(1-exp(-c))*(1.-exp(-c*(((Hmid*(1-rho/rho_w)-0)/(2000.0-0))))))*(1-grounded)

# Linear SMB, will be replaced by Orographic Precip Model
Smax = 1500.  # above Smax, adot=amax [m]
Smin = 0.     # below Smin, adot=amin [m]

# For testing, we can define SMB as an expression:
adot = amin + (amax - amin) / (Smax - Smin) * (S * grounded + Hmid * (1 - rho / rho_w) * (1 - grounded))

# SMB function interpolate from an array as provided by the Orographic Precipitation Model
test_H = h_max * np.exp(-(((x-x0)**2/(2*sigma_x**2)))) + zmin
smb =   amin + (amax - amin) / (Smax - Smin) * test_H
adot_f = function_from_array(x, smb, Q, mesh)

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

vi = VerticalIntegrator(points,weights)

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
R += ((H-H0)/dt*xsi  - xsi.dx(0)*U[0]*Hmid + D*xsi.dx(0)*Hmid.dx(0) - (adot - un*H0/width*width.dx(0))*xsi)*dx  + U[0]*area*xsi*ds(1)

# Jacobian of coupled momentum-mass system
J = derivative(R,U,dU)

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
assigner_inv = FunctionAssigner([Q,Q,Q],V)
assigner     = FunctionAssigner(V,[Q,Q,Q])

#####################################################################
######################  Variational Solvers  ########################
#####################################################################

# Define variational solver for the momentum problem

# Ice divide dirichlet bc
bc = DirichletBC(V.sub(2),thklim,lambda x,on: near(x[0],-L) and on)

# No Dirichlet BCs for symmetric geometry, both sides are ocean
# mass_problem = NonlinearVariationalProblem(R,U,bcs=[bc],J=J,form_compiler_parameters=ffc_options)
mass_problem = NonlinearVariationalProblem(R,U,J=J,form_compiler_parameters=ffc_options)

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

################## PLOTTING ##########################
# ion()
# fig,ax = subplots(nrows=2,sharex=True)
x = mesh.coordinates().ravel()
SS = project(S)
BB = B.compute_vertex_values()
# ph0, = ax[0].plot(x,BB,'b-')

HH = H0.compute_vertex_values()

# ph1, = ax[0].plot(x,BB+HH,'g-')
# ph5, = ax[0].plot(x,BB+HH,'r-')
# ax[0].set_xlim(-L,L/2.)
# ax[0].set_ylim(-1000,2500)

us = project(u(0))
ub = project(u(1))
# ph3, = ax[1].plot(x,us.compute_vertex_values())
# ph4, = ax[1].plot(x,ub.compute_vertex_values())

# ax[1].set_xlim(-L,L/2.)
# ax[1].set_ylim(0,400)

# draw()
mass = []
time = []

tdata = []
Hdata = []
hdata = []
Bdata = []
usdata = []
ubdata = []
gldata = []
grdata = []

######################################################################
#######################   SOLUTION   #################################
######################################################################

# Time interval
t = 0.0
t_end = 50.
dt_float = 0.5             # Set time step here
dt.assign(dt_float)

assigner.assign(U,[ze,ze,H0])

# Loop over time
while t<t_end:
    time.append(t)

    # Update grounding line position
    solve(A_g == b_g,grounded)
    grounded.vector()[0] = 1

    # Try solving with last solution as initial guess for next solution
    try:
        mass_solver.solve(l_bound,u_bound)
    # If this breaks, set initial guess to zero and try again
    except:
        assigner.assign(U,[ze,ze,H0])
        mass_solver.solve(l_bound,u_bound)

    # Set previous time step variables
    assigner_inv.assign([un,u2n,H0],U)

    # Plotting
    BB = B.compute_vertex_values()
    HH = H0.compute_vertex_values()

    us = project(u(0))
    ub = project(u(1))

    # ph0.set_ydata(BB)
    # ph1.set_ydata((BB + HH)*grounded.compute_vertex_values() + (1-rho/rho_w)*HH*(1-grounded.compute_vertex_values()))
    # ph5.set_ydata((BB)*grounded.compute_vertex_values() + (-rho/rho_w*HH)*(1-grounded.compute_vertex_values()))

    # ph3.set_ydata(us.compute_vertex_values())
    # ph4.set_ydata(ub.compute_vertex_values())
    # draw()

    # Save values at each time step
    tdata.append(t)
    Hdata.append(H0.vector().array())
    Bdata.append(B.vector().array())
    gldata.append(gl(0))
    usdata.append(us.vector().array())
    ubdata.append(ub.vector().array())
    grdata.append(grounded.vector().array())
    
    print t,H0.vector().max()
    t+=dt_float

# Save relevant data to pickle
pickle.dump((tdata,Hdata,hdata,Bdata,usdata,ubdata,grdata,gldata),open('good_filename_here.p','w'))

