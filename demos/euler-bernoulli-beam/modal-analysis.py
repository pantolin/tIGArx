"""
Solve and plot several modes of the cantilevered Euler--Bernoulli beam,
using a pure displacement formulation, which would not be possible with
standard $C^0$ finite elements.

Note: This demo uses interactive plotting, which may cause errors on systems
without GUIs.
"""
import numpy as np
import matplotlib.pyplot as plt

from petsc4py import PETSc
from slepc4py import SLEPc

from dolfinx import default_real_type
import ufl

from tIGArx.common import mpisize
from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots

from tIGArx.ExtractedSpline import ExtractedSpline
from tIGArx.MultiFieldSplines import EqualOrderSpline


if mpisize > 1:
    print("ERROR: This demo only works in serial. "
          "SLEPc fails in parallel. Possibly related to "
          "https://slepc.upv.es/documentation/faq.htm#faq10 ?")
    exit()


####### Preprocessing #######

# Polynomial degree of the basis functions: must be >1 for this demo, because
# functions need to be at least $C^1$ for the formulation.
p = 3

# Number of elements to divide the beam into:
Nel = 100

# Length of the beam:
L = 1.0

# Create a univariate B-spline.
splineMesh = ExplicitBSplineControlMesh(
    [
        p,
    ],
    [
        uniform_knots(p, 0.0, L, Nel),
    ],
)
splineGenerator = EqualOrderSpline(1, splineMesh)

# Apply Dirichlet BCs to the first two nodes, for a clamped BC.
field = 0
parametricDirection = 0
side = 0
scalarSpline = splineGenerator.getScalarSpline(field)
sideDofs = scalarSpline.getSideDofs(parametricDirection, side, nLayers=2)
splineGenerator.addZeroDofs(field, sideDofs)


####### Analysis #######

QUAD_DEG = 2 * p
spline = ExtractedSpline(splineGenerator, QUAD_DEG)

# Displacement test and trial functions:
u = ufl.TrialFunction(spline.V)
v = ufl.TestFunction(spline.V)


# Laplace operator:
def lap(f):
    return spline.div(spline.grad(f))


# Material constants for the Euler--Bernoulli beam problem:
E = 1.0
I = 1.0
mu = 1.0

# Elasticity form:
a = ufl.inner(E * I * lap(u), lap(v)) * spline.dx

# Mass form:
b = mu * ufl.inner(u, v) * spline.dx

# Assemble the matrices for a generalized eigenvalue problem.  The reason that
# the diagonal entries for A corresponding to Dirichlet BCs are set to a
# large value is to shift the corresponding eigenmodes to the high end of
# the frequency spectrum.
A = spline.assembleMatrix(a, diag=1.0 / np.finfo(default_real_type).eps)
B = spline.assembleMatrix(b)

# Solve the eigenvalue problem, ordering values from smallest to largest in
# magnitude.

comm = spline.mesh.comm
mpi_rank = comm.Get_rank()

N_MODES = 5

opts = PETSc.Options()
opts.setValue("eps_target_magnitude", None)
opts.setValue("eps_target", 0)
opts.setValue("st_type", "sinvert")

eig_solver = SLEPc.EPS().create(comm)
eig_solver.setDimensions(N_MODES)
eig_solver.setOperators(A, B)
eig_solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
eig_solver.setFromOptions()
eig_solver.solve()

assert N_MODES < eig_solver.getConverged()

if mpi_rank == 0:
    its = eig_solver.getIterationNumber()
    print("Number of iterations of the method: %d" % its)

    eps_type = eig_solver.getType()
    print("Solution method: %s" % eps_type)

    nev, ncv, mpd = eig_solver.getDimensions()
    print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = eig_solver.getTolerances()
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

size_local = spline.V.dofmap.index_map.size_local
x = spline.V.tabulate_dof_coordinates()[:size_local, 0]
x_global = comm.gather(x, root=0)
if mpi_rank == 0:
    x_global = np.concatenate(x_global)
    ind = np.argsort(x_global)
    x_global = x_global[ind]


# Look at the first N_MODES modes of the problem.
for n in range(0, N_MODES):
    # Due to the structure of the problem, we know that the eigenvalues are
    # real, so we are passing the dummy placeholder _ for the complex parts
    # of the eigenvalue and mode.

    uVectorIGA = A.getVecLeft()
    omega2 = eig_solver.getEigenpair(n, Vr=uVectorIGA)
    if mpi_rank == 0:
        print("omega_" + str(n) + " = " + str(np.sqrt(omega2)))

    # The solution from the eigensolver is a vector of IGA DoFs, and must be
    # extracted back to an FE representation for plotting.
    u = spline.M * uVectorIGA

    u_global = comm.gather(u.array_r, root=0)

    if mpi_rank == 0:
        u_global = np.concatenate(u_global)[ind]
        plt.plot(x_global, u_global)

if mpi_rank == 0:
    plt.autoscale()
    plt.show()
