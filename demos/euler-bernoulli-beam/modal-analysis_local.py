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

import ufl
import dolfinx

from tigarx.LocalAssembly import assemble_matrix
from tigarx.LocalSpline import LocallyConstructedSpline
from tigarx.common import mpisize
from tigarx.BSplines import ExplicitBSplineControlMesh, uniform_knots


def modal_analysis_local():

    if mpisize > 1:
        print("ERROR: This demo only works in serial. "
              "SLEPc fails in parallel. Possibly related to "
              "https://slepc.upv.es/documentation/faq.htm#faq10 ?")
        exit()

    # Since there can only be one rank here, no need for mpi_rank == 0 checks

    p = 3
    Nel = 100
    L = 1.0

    # Material constants for the Euler--Bernoulli beam problem:
    E = 1.0
    I = 1.0
    mu = 1.0

    # Create a univariate B-spline.
    spline_mesh = ExplicitBSplineControlMesh(
        [p], [uniform_knots(p, 0.0, L, Nel)]
    )

    scalar_spline = spline_mesh.getScalarSpline()

    spline = LocallyConstructedSpline.get_from_mesh_and_init(
        spline_mesh, quad_degree=2 * p, dofs_per_cp=1
    )

    side_dofs = np.array(
        np.unique(scalar_spline.getSideDofs(0, 0, layers=2)),
        dtype=np.int32
    )

    # Displacement test and trial functions:
    u = ufl.TrialFunction(spline.V)
    v = ufl.TestFunction(spline.V)

    # Laplace operator:
    def lap(f):
        return spline.div(spline.grad(f))

    # Elasticity form:
    a = ufl.inner(E * I * lap(u), lap(v)) * spline.dx

    # Mass form:
    b = mu * ufl.inner(u, v) * spline.dx

    A = assemble_matrix(dolfinx.fem.form(a), scalar_spline)
    # Make locked dofs very stiff to enforce Dirichlet BCs.
    A.zeroRowsColumns(side_dofs, 1.0e12)

    M = assemble_matrix(dolfinx.fem.form(b), scalar_spline)
    # Mass of locked dofs is fixed to fix their ratio to the stiffness.
    M.zeroRowsColumns(side_dofs, 1.0)

    # Solve the eigenvalue problem, ordering values from smallest to largest in
    # magnitude.

    comm = spline.mesh.comm

    n_modes = 5

    opts = PETSc.Options()
    opts.setValue("eps_target_magnitude", None)
    opts.setValue("eps_target", 0)
    opts.setValue("st_type", "sinvert")

    eig_solver = SLEPc.EPS().create(comm)
    eig_solver.setDimensions(n_modes)
    eig_solver.setOperators(A, M)
    eig_solver.setProblemType(SLEPc.EPS.ProblemType.GHEP)
    eig_solver.setFromOptions()
    eig_solver.solve()

    assert n_modes < eig_solver.getConverged()

    its = eig_solver.getIterationNumber()
    print("Number of iterations of the method: %d" % its)

    eps_type = eig_solver.getType()
    print("Solution method: %s" % eps_type)

    nev, ncv, mpd = eig_solver.getDimensions()
    print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = eig_solver.getTolerances()
    print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    # The second coordinate is for NURBS, but this is a B-spline -> ignore
    x = spline.extracted_control_points[:, 0]
    u = dolfinx.fem.Function(spline.V)

    # Look at the first N_MODES modes of the problem.
    for n in range(0, n_modes):
        # Due to the structure of the problem, we know that the eigenvalues are
        # real, so we are passing the dummy placeholder _ for the complex parts
        # of the eigenvalue and mode.

        u_vector_iga = A.getVecLeft()
        omega2 = eig_solver.getEigenpair(n, Vr=u_vector_iga)

        print("omega_" + str(n + 1) + " = " + str(np.sqrt(omega2)))

        # The solution from the eigensolver is a vector of IGA DoFs, and must be
        # extracted back to an FE representation for plotting.
        spline.extract_cp_solution_to_fe(u_vector_iga, u)

        plt.plot(x, u.x.array)

    plt.autoscale()
    plt.show()


if __name__ == "__main__":
    modal_analysis_local()
