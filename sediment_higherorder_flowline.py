####################################################################################
##########################  sediment_ho_flowline.py  ###############################
####################################################################################

# Author: Douglas Brinkerhoff, 2021
# License: GNU GPLv3`
# Requires Python3 and libraries: matplotlib, fenics 2019.1, numpy
####################################################################################
####################################################################################
####################################################################################

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import matplotlib
import dolfin as df
import ufl
import matplotlib.pyplot as plt
import numpy as np

df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 2
df.parameters['allow_extrapolation'] = True

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.description = "Variational Inference of PDD parameters."
parser.add_argument(
    "-g",
    "--geometry",
    dest="geometry",
    choices=["1sided", "sym", "asym"],
    help="Geometry",
    default="1sided",
)
options = parser.parse_args()
geom = options.geometry


##########################################################
###############        CONSTANTS       ###################
##########################################################
# HELPER FUNCTIONS
def Max(a, b):
    return (a + b + abs(a - b)) / df.Constant(2)


def Min(a, b):
    return (a + b - abs(a - b)) / df.Constant(2)


def softplus(y1, y2, alpha=1):
    # The softplus function is a differentiable approximation
    # to the ramp function.  Its derivative is the logistic function.
    # Larger alpha makes a sharper transition.
    return Max(y1, y2) + (1.0 / alpha) * df.ln(
        1 + df.exp(alpha * (Min(y1, y2) - Max(y1, y2)))
    )


from numpy.polynomial.legendre import leggauss


def full_quad(order):
    # This function provides the points and weights for
    # Gaussian quadrature (here used in the vertical dimension)
    points, weights = leggauss(order)
    points = (points + 1) / 2.0
    weights /= 2.0
    return points, weights


# Logistic function
sigmoid = lambda z: 1.0 / (1 + df.exp(-z))

spy = 31556925.9747  # seconds per year [s year-1]
thklim = 1.0  # Minimum Ice Thickness

c = 2.0  # Coefficient of exponential decay

amp = 100.0  # Amplitude of sinusoidal topography

rho = rho_i = 917.0  # Ice density
rho_w = 1029.0  # Seawater density
rho_s = 1600.0  # Sediment density
rho_r = 2750.0  # Bedrock density

La = 3.35e5

g = 9.81  # Gravitational acceleration
n = 3.0  # Glen's exponent
m = 1.0  # Sliding law exponent
b = 1e-16 ** (-1.0 / n)  # Ice hardness
eps_reg = df.Constant(1e-4)  # Regularization parameter

l_s = df.Constant(2.0)  # Sediment thickness at which bedrock erosion becomes negligible
be = df.Constant(1e-8)  # Bedrock erosion coefficient
cc = df.Constant(2e-11)  # Fluvial erosion coefficient
d = df.Constant(500.0)  # Fallout fraction
h_0 = df.Constant(0.1)  # Subglacial cavity depth


k = df.Constant(0.7)

dt_float = 0.1  # Time step
dt = df.Constant(dt_float)

if geom == "1sided":
    L = 45000.0  # Characteristic domain length
    zmin = -300.0  # Minimum elevation
    zmax = 2200.0  # Maximum elevation
    amin = df.Constant(-7.0)  # Minimum smb
    amax = df.Constant(5.0)  # Maximum smb
else:
    L = 75000.0  # Characteristic domain length
    zmin = -500.0  # Minimum elevation
    zmax = 2500.0  # Maximum elevation
    amin = df.Constant(-8.0)  # Minimum smb
    amax = df.Constant(10.0)  # Maximum smb

#########################################################
#################      GEOMETRY     #####################
#########################################################

# Topography
my_dx = 1000.0  # [m]
x = np.arange(-L, L + my_dx, my_dx)  # [m]

x0 = 0
sigma_x = 15e3
sigma_x1 = 25e3
sigma_x2 = 10e3


# Bed elevation
class Bed1Sided(df.UserExpression):
    def eval(self, values, x):
        values[0] = (
            (zmax - zmin) * np.exp(-(x[0] + L) / (L * 0.5))
            + zmin
            - amp * (np.sin(4 * np.pi * x[0] / L))
        )


# Bed elevation Expression
class BedSym(df.UserExpression):
    def eval(self, values, x):
        values[0] = zmax * np.exp(-(((x[0] - x0) ** 2 / (2 * sigma_x**2)))) + zmin


class BedAsym(df.UserExpression):
    def eval(self, values, x):
        values[0] = (
            zmax
            * ufl.conditional(
                ufl.gt(x[0], 0),
                np.exp(-(((x[0] - x0) ** 2 / (2 * sigma_x1**2)))),
                np.exp(-(((x[0] - x0) ** 2 / (2 * sigma_x2**2)))),
            )
            + zmin
        )

class FlowDir1Sided(df.UserExpression):
    def eval(self,values,x):
        values[0] = 1.0

class FlowDirSym(df.UserExpression):
    def eval(self,values,x):
        if x[0]>0:
            values[0] = 1.0
        else:
            values[0] = -1.0

# Basal traction
class Beta2(df.UserExpression):
    def eval(self, values, x):
        values[0] = 50


##########################################################
################           MESH          #################
##########################################################

# Define a rectangular mesh
nx = 500
mesh = df.IntervalMesh(nx, -L, L)

# Define boundaries
ocean = df.MeshFunction("size_t", mesh, 1, 0)
ds = df.ds(subdomain_data=ocean)

for f in df.facets(mesh):
    if df.near(f.midpoint().x(), L):
        ocean[f] = 1
    if df.near(f.midpoint().x(), -L):
        if geom in "1sided":
            ocean[f] = 2
        else:
            ocean[f] = 3

#########################################################
#################  FUNCTION SPACES  #####################
#########################################################

nhat = df.FacetNormal(mesh)[0]

# CG1 Function Space
E_cg = df.FiniteElement("CG", mesh.ufl_cell(), 1)
Q_cg = df.FunctionSpace(mesh, E_cg)

# DG0 Function Space
E_dg = df.FiniteElement("DG", mesh.ufl_cell(), 0)
Q_dg = df.FunctionSpace(mesh, E_dg)

# Mixed element for coupled velocity-thickness solve
# (depth-averaged velocity, deformational velocity, DG0 thickness, CG1 thickness projection)
E_glac = df.MixedElement([E_cg, E_cg, E_dg, E_cg])
V_g = df.FunctionSpace(mesh, E_glac)

# Mixed element for coupled sediment stuff
# (Bedrock elevation, fluvial sediment flux, sediment thickness,
# CG1 sediment thickness projection, effective subglacial cavity height
E_sed = df.MixedElement([E_cg, E_dg, E_dg, E_cg, E_dg])
V_sed = df.FunctionSpace(mesh, E_sed)

zero_cg = df.Function(Q_cg)

#########################################################
#################  FUNCTIONS  ###########################
#########################################################

# Velocity and thickness functions
U = df.Function(V_g)
dU = df.TrialFunction(V_g)
Phi = df.TestFunction(V_g)

# Split into components
ubar, udef, H, H_ = df.split(U)
phibar, phidef, xsi, w = df.split(Phi)

# Sediment functions
T = df.Function(V_sed)
dT = df.TrialFunction(V_sed)
Psi = df.TestFunction(V_sed)

# Split into components
B, Qs, h_s, h_s_, h_eff = df.split(T)
psi_B, psi_Q, psi_h, psi_h_, psi_eff = df.split(Psi)

# Functions to hold results from previous time step
ubar0 = df.Function(Q_cg)
udef0 = df.Function(Q_cg)

H0 = df.Function(Q_dg)
H0_ = df.Function(Q_cg)

H0.vector()[:] = 25
H0_.vector()[:] = 25

if geom in "sym":
    Bed = BedSym
    FlowDir = FlowDirSym
elif geom in "asym":
    Bed = BedAsym
    FlowDir = FlowDirSym
elif geom in "1sided":
    Bed = Bed1Sided
    FlowDir = FlowDir1Sided
else:
    print(("{} not supported".format(geom)))


B0 = df.interpolate(Bed(), Q_cg)
B0_ = df.interpolate(Bed(), Q_dg)

flow_dir = df.interpolate(FlowDir(), Q_dg)

Qs0 = df.Function(Q_dg)

h_s0 = df.Function(Q_dg)
h_s_0 = df.Function(Q_cg)

h_eff0 = df.Function(Q_dg)
h_eff0.vector()[:] = h_0(0)

# Scalar test functions for uncoupled water flux
psi = df.TestFunction(Q_dg)
dQ = df.TrialFunction(Q_dg)
ww = df.TestFunction(Q_cg)


# Functions for computing the grounded indicator
grounded = df.Function(Q_dg)
grounded.vector()[:] = 1
dg = df.TrialFunction(Q_dg)

ghat = df.Function(Q_cg)
gl = df.Constant(0)


Bhat = B + h_s_

l = softplus(df.Constant(0), Bhat)  # Water surface, or the greater of
# bedrock topography or zero

Base = softplus(Bhat, -rho / rho_w * H_, alpha=1.0)  # Ice base is the greater of the
# bedrock topography or the base of
# the shelf

D = softplus(-Bhat, df.Constant(0))  # Water depth
S = Base + H_

ghat = 1 / (1 + df.exp(-(H * rho_i / rho_w + 3 - D)))  # Approximate flotation indicator

beta2 = df.interpolate(Beta2(), Q_cg)  # Traction


Smax = 2500.0  # above Smax, adot=amax [m]
Smin = 200.0  # below Smin, adot=amin [m]
Sela = 1000.0  # equilibrium line altidue [m]

if geom == "1sided":
    climate_factor = df.Constant(1.0)  # Climate
    adot = climate_factor * (
        amin + (amax - amin) / (1 - df.exp(-c)) * (1.0 - df.exp(-c * ((S / 2000))))
    )  # *grounded + (-0.5*H)*(1-grounded)
    bdot = df.Constant(0)
else:
    adot = ufl.conditional(
        ufl.lt(S, Sela),
        (-amin / (Sela - Smin)) * (S - Sela),
        (amax / (Smax - Sela)) * (S - Sela),
    ) * grounded + ufl.conditional(
        ufl.lt(S, Sela),
        (-amin / (Sela - Smin)) * (H0 - Sela),
        (amax / (Smax - Sela)) * (H0 * (1 - rho / rho_w) - Sela),
    ) * (
        1 - grounded
    )
    bdot = df.Constant(0.0) * (1 - grounded)


########################################################
#################   MOMENTUM BALANCE   #################
########################################################


class VerticalBasis(object):
    def __init__(self, u, coef, dcoef):
        self.u = u
        self.coef = coef
        self.dcoef = dcoef

    def __call__(self, s):
        return sum([u * c(s) for u, c in zip(self.u, self.coef)])

    def ds(self, s):
        return sum([u * c(s) for u, c in zip(self.u, self.dcoef)])

    def dx(self, s, x):
        return sum([u.dx(x) * c(s) for u, c in zip(self.u, self.coef)])


class VerticalIntegrator(object):
    def __init__(self, points, weights):
        self.points = points
        self.weights = weights

    def integral_term(self, f, s, w):
        return w * f(s)

    def intz(self, f):
        return sum(
            [self.integral_term(f, s, w) for s, w in zip(self.points, self.weights)]
        )


def dsdx(s):
    return 1.0 / H_ * (S.dx(0) - s * H_.dx(0))


def dsdz(s):
    return -1.0 / H_


# ANSATZ
p = 4.0
coef = [lambda s: 1.0, lambda s: 1.0 / p * ((p + 1) * s**p - 1.0)]
dcoef = [lambda s: 0.0, lambda s: (p + 1) * s ** (p - 1)]

u_ = [ubar, udef]
u0_ = [ubar0, udef0]
phi_ = [phibar, phidef]

u = VerticalBasis(u_, coef, dcoef)
phi = VerticalBasis(phi_, coef, dcoef)


def eta_v(s):
    return (
        b
        / 2.0
        * (
            (u.dx(s, 0) + u.ds(s) * dsdx(s)) ** 2
            + 0.25 * ((u.ds(s) * dsdz(s)) ** 2)
            + eps_reg
        )
        ** ((1.0 - n) / (2 * n))
    )


def membrane_xx(s):
    return (
        (phi.dx(s, 0) + phi.ds(s) * dsdx(s))
        * H_
        * eta_v(s)
        * (4 * (u.dx(s, 0) + u.ds(s) * dsdx(s)))
    )


def shear_xz(s):
    return dsdz(s) ** 2 * phi.ds(s) * H_ * eta_v(s) * u.ds(s)


def tau_dx():
    return rho * g * H_ * S.dx(0) * phibar


points, weights = full_quad(4)

vi = VerticalIntegrator(points, weights)

# Pressure and sliding law
P_0 = H
P_w = ufl.Max(k * H, rho_w / rho_i * (l - Base))
N = ufl.Max(P_0 - P_w, df.Constant(0.000))

I_stress = (
    -vi.intz(membrane_xx) - vi.intz(shear_xz) - phi(1) * beta2 * u(1) * N - tau_dx()
) * df.dx

#############################################################################
##########################  MASS BALANCE  ###################################
#############################################################################


H_avg = 0.5 * (H("+") + H("-"))
H_jump = H("+") * nhat("+") + H("-") * nhat("-")
xsi_avg = 0.5 * (xsi("+") + xsi("-"))
xsi_jump = xsi("+") * nhat("+") + xsi("-") * nhat("-")

uvec = df.as_vector(
    [
        ubar,
    ]
)
unorm = (df.dot(uvec, uvec)) ** 0.5
uH = df.avg(ubar) * H_avg + 0.5 * df.avg(unorm) * H_jump

if geom == '1sided':
    I_transport = (
        ((H - H0) / dt - (adot + bdot)) * xsi * df.dx
        + df.dot(uH, xsi_jump) * df.dS
        + ubar * H * nhat * xsi * ds(1)
    )
else:
    I_transport = (
        ((H - H0) / dt - (adot + bdot)) * xsi * df.dx
        + df.dot(uH, xsi_jump) * df.dS
        + ubar * H * nhat * xsi * ds#(1)
    )

# This projects the DG0 thickness onto a CG1 space, so that we can take derivatives
I_project = (H - H_) * w * df.dx

# Weak form of coupled velocity/thickness solve
R = I_stress + I_transport + I_project

J = df.derivative(R, U, dU)

#############################################################################
###########################  Water Flux  ####################################
#############################################################################

# Meltrate
me = (beta2 * N * u(1) ** 2 / (rho * La) - Min(adot, -1e-16)) * sigmoid(
    H - (thklim + df.Constant(1))
)
# h = df.CellDiameter(mesh)

dQ_avg = 0.5 * (dQ("+") + dQ("-"))
dQ_jump = dQ("+") * nhat("+") + dQ("-") * nhat("-")
psi_avg = 0.5 * (psi("+") + psi("-"))
psi_jump = psi("+") * nhat("+") + psi("-") * nhat("-")

dQ_upwind = df.avg(flow_dir)*dQ_avg + 0.5 * dQ_jump

Qw = df.Function(Q_dg)

if geom=='1sided':
    W_div = (
    -me * psi * df.dx + df.dot(dQ_upwind, psi_jump) * df.dS + dQ * flow_dir * nhat * psi * ds(1)
    )
else:
    W_div = (
    -me * psi * df.dx + df.dot(dQ_upwind, psi_jump) * df.dS + dQ * flow_dir * nhat * psi * ds#(1)
    )

R_Qw = W_div
A_Qw = df.lhs(R_Qw)
b_Qw = df.rhs(R_Qw)

#############################################################################
#############################  Sediment evolution  ##########################
#############################################################################


delta = df.exp(-h_s / l_s)
# average water speed is equal to (water flux) / (effective thickness) (Eq 4)
ubar_w = Qw / h_eff

# Rate of bedrock erosion (Eq 2, RHS)
Bdot = -be * beta2 * N * u(1) ** 2 * delta
# Erosion rate (Eq 6)
edot = cc / h_eff * ubar_w**2 * (1 - delta)
# Deposition rate (Eq 7)
ddot = d * Qs / Qw

Qs_avg = 0.5 * (Qs("+") + Qs("-"))
Qs_jump = Qs("+") * nhat("+") + Qs("-") * nhat("-")
psiQ_avg = 0.5 * (psi_Q("+") + psi_Q("-"))
psiQ_jump = psi_Q("+") * nhat("+") + psi_Q("-") * nhat("-")

# diffusivity of sediment due to hill-slope processes
k_diff = df.Constant(5000)

psih_avg = 0.5 * (psi_h("+") + psi_h("-"))
psih_jump = psi_h("+") * nhat("+") + psi_h("-") * nhat("-")

Qs_upwind = df.avg(flow_dir)*Qs_avg + 0.5 * Qs_jump
# Sediment flux (Eq 8)
if geom == '1sided':
    R_Qs = (
        (ddot - edot) * psi_Q * df.dx
        + df.dot(Qs_upwind, psiQ_jump) * df.dS
        + Qs * flow_dir * nhat * psi_Q * ds(1)
    )
else:
    R_Qs = (
        (ddot - edot) * psi_Q * df.dx
        + df.dot(Qs_upwind, psiQ_jump) * df.dS
        + Qs * flow_dir * nhat * psi_Q * ds#(1)
    )
# Sediment transport (Eq 5)
h = df.CellDiameter(mesh)
dhsdt = (h_s("+") - h_s("-")) / (0.5 * (h("+") + h("-")))
R_hs = (
    psi_h * ((h_s - h_s0) / dt + rho_r / rho_s * Bdot - ddot + edot) * df.dx
    + df.avg(k_diff) * dhsdt * psih_jump * df.dS
)
# Bedrock evolution (Eq 2)
R_B = psi_B * ((B - B0) / dt - Bdot) * df.dx
# ??
R_hsx = psi_h_ * (h_s - h_s_) * df.dx
# Effective thickness ?
R_heff = psi_eff * (h_eff - softplus(h_0, Base - Bhat, alpha=10.0)) * df.dx

# Weak form of sediment dynamics, solves for bedrock elevation, fluvial sed. flux,
# sediment thickness, projected sediment thickness, and effective water layer thickness.
R_sed = R_B + R_Qs + R_hs + R_hsx + R_heff
J_sed = df.derivative(R_sed, T, dT)
#####################################################################
#########################  I/O Functions  ###########################
#####################################################################

# For moving data between vector functions and scalar functions
assigner_inv_g = df.FunctionAssigner([Q_cg, Q_cg, Q_dg, Q_cg], V_g)
assigner_g = df.FunctionAssigner(V_g, [Q_cg, Q_cg, Q_dg, Q_cg])

assigner_inv_s = df.FunctionAssigner([Q_cg, Q_dg, Q_dg, Q_cg, Q_dg], V_sed)
assigner_s = df.FunctionAssigner(V_sed, [Q_cg, Q_dg, Q_dg, Q_cg, Q_dg])

#####################################################################
######################  Variational Solvers  ########################
#####################################################################

# Bounds
l_thick_bound = df.project(df.Constant(thklim), Q_dg)
u_thick_bound = df.project(df.Constant(1e4), Q_dg)

l_thick_bound_ = df.project(df.Constant(thklim), Q_cg)
u_thick_bound_ = df.project(df.Constant(1e4), Q_cg)

l_v_bound = df.project(-100000.0, Q_cg)
u_v_bound = df.project(100000.0, Q_cg)

l_bound = df.Function(V_g)
u_bound = df.Function(V_g)

assigner_g.assign(l_bound, [l_v_bound] * 2 + [l_thick_bound] + [l_thick_bound_])
assigner_g.assign(u_bound, [u_v_bound] * 2 + [u_thick_bound] + [u_thick_bound_])

# Define variational solver for the momentum problem
mass_problem = df.NonlinearVariationalProblem(R, U, bcs=[], J=J)
mass_problem.set_bounds(l_bound, u_bound)

sed_problem = df.NonlinearVariationalProblem(R_sed, T, J=J_sed)

######################################################################
#######################   SOLUTION   #################################
######################################################################


### PLOTTING ###
plt.ion()
fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(10, 12))

x = mesh.coordinates()
BB = B0.compute_vertex_values()
SS = df.project(S).compute_vertex_values()
Ba = df.project(Base).compute_vertex_values()
(ph_bed,) = ax[0].plot(x, BB, "k-", lw=2.0)
(ph_surface,) = ax[0].plot(x, SS, "c-", lw=1.0)
(ph_bottom,) = ax[0].plot(x, Ba, "c-", lw=1.0)
(ph_sed,) = ax[0].plot(x, SS, "g-", lw=1.0)
ax[0].plot(x, np.zeros_like(x), "b:", lw=1.0)
ax[0].set_ylabel("Elevation")
ax[0].set_ylim(-500, 3000)

(ph_v,) = ax[1].plot(x, np.zeros_like(x), "k-", lw=1.0)
ax[1].set_ylabel("Water flux")

(ph_us,) = ax[2].plot(x, np.zeros_like(x), "r-", lw=1.0)
(ph_ub,) = ax[2].plot(x, np.zeros_like(x), "k-", lw=1.0)
ax[2].set_ylim(0, 500)
ax[2].set_ylabel("Abs(Speed) (m/a)")

(ph_hs,) = ax[3].plot(x, np.zeros_like(x), "k-", lw=1.0)
ax[3].set_ylabel("Sed. Thk.")
ax[3].set_xlabel("Dist.")

plt.pause(0.00001)

# Time interval
t = 0.0
t_end = 10000.0

counter = 0

# Maximum time step!!  Increase with caution.
dt_max = 1.0

# Initialization stuff
ubarinit = df.Function(Q_cg)
udefinit = df.Function(Q_cg)
ubarinit.vector()[:] += (
    1e-1 * np.random.randn(ubar0.vector().get_local().shape[0]) + 100.0
)
udefinit.vector()[:] += 1e-3 * np.random.randn(udef0.vector().get_local().shape[0])
ubar0.vector()[:] += 1e-1 * np.random.randn(ubar0.vector().get_local().shape[0])
udef0.vector()[:] += 1e-3 * np.random.randn(udef0.vector().get_local().shape[0])
assigner_g.assign(U, [ubar0, udef0, H0, H0_])
assigner_s.assign(T, [B0, Qs0, h_s0, h_s_0, h_eff0])

# Loop over time
while t < t_end:
    try:  # If the solvers don't converge, reduce the time step and try again.
        print(t, dt_float, H0.vector().max(), df.assemble(h_s0 * df.dx))

        assigner_s.assign(T, [B0, Qs0, h_s0, h_s_0, h_eff0])
        assigner_g.assign(U, [ubar0, udef0, H0, H0_])

        # Solve for water flux
        df.solve(A_Qw == b_Qw, Qw)

        # Solve for sediment variables
        print("solving sed")
        sed_solver = df.NonlinearVariationalSolver(sed_problem)
        sed_solver.parameters["nonlinear_solver"] = "newton"

        sed_solver.parameters["newton_solver"]["relative_tolerance"] = 1e-2
        sed_solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-2
        sed_solver.parameters["newton_solver"]["error_on_nonconvergence"] = True
        sed_solver.parameters["newton_solver"]["linear_solver"] = "gmres"
        sed_solver.parameters["newton_solver"]["maximum_iterations"] = 10
        sed_solver.parameters["newton_solver"]["report"] = True
        sed_solver.parameters["newton_solver"]["relaxation_parameter"] = 0.7
        sed_solver.parameters['newton_solver']['krylov_solver']['relative_tolerance'] = 1e-3

        sed_solver.solve()

        # Solve for ice velocity and thickness
        print("solving mass")
        assigner_g.assign(U, [ubarinit, zero_cg, H0, H0_])
        mass_solver = df.NonlinearVariationalSolver(mass_problem)
        mass_solver.parameters["nonlinear_solver"] = "snes"

        mass_solver.parameters["snes_solver"]["method"] = "vinewtonrsls"
        mass_solver.parameters["snes_solver"]["relative_tolerance"] = 1e-2
        mass_solver.parameters["snes_solver"]["absolute_tolerance"] = 1e-2
        mass_solver.parameters["snes_solver"]["error_on_nonconvergence"] = True
        mass_solver.parameters["snes_solver"]["linear_solver"] = "gmres"
        mass_solver.parameters["snes_solver"]["maximum_iterations"] = 10
        mass_solver.parameters["snes_solver"]["report"] = True
        mass_solver.parameters['snes_solver']['krylov_solver']['relative_tolerance'] = 1e-3
        mass_solver.solve()

        assigner_inv_s.assign([B0, Qs0, h_s0, h_s_0, h_eff0], T)
        assigner_inv_g.assign([ubar0, udef0, H0, H0_], U)

        # Increase time step if solvers complete successfully
        dt_float = min(1.05 * dt_float, dt_max)
        dt.assign(dt_float)

        # Do some plotting
        thk = H0_.compute_vertex_values()
        bed = B0.compute_vertex_values()
        surface = df.project(S).compute_vertex_values()
        surface[thk <= (thklim + 1e-2)] = np.nan
        bottom = df.project(Base).compute_vertex_values()
        bottom[thk <= (thklim + 1e-2)] = np.nan

        grounded = df.project(ghat, Q_dg)
        g_ = grounded.compute_vertex_values()
        sed_surface = bed + h_s0.compute_vertex_values()

        ph_bed.set_ydata(bed)
        ph_surface.set_ydata(surface)
        ph_bottom.set_ydata(bottom)
        ph_sed.set_ydata(sed_surface)

        ph_v.set_ydata(df.project(Qw).compute_vertex_values())
        ax[1].set_ylim(
            -df.project(Qw).compute_vertex_values().max(),
            df.project(Qw).compute_vertex_values().max(),
        )
        us_ = df.project(u(0)).compute_vertex_values()
        ub_ = df.project(u(1)).compute_vertex_values()
        us_[thk <= (thklim + 1e-2)] = np.nan
        ub_[thk <= (thklim + 1e-2)] = np.nan

        ph_us.set_ydata(np.abs(us_))
        ph_ub.set_ydata(np.abs(ub_))

        ph_hs.set_ydata(h_s0.compute_vertex_values())
        ax[3].set_ylim(0, h_s0.compute_vertex_values().max() + 10)

        if counter % 10 == 0:
            # pause(0.00001)
            fig.canvas.start_event_loop(0.001)
            fig.canvas.draw_idle()

        t += dt_float
        counter += 1
    except RuntimeError:
        dt_float /= 2.0
        dt.assign(dt_float)
        print("convergence failed, reducing time step and trying again")
