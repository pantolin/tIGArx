import numpy as np

from petsc4py import PETSc
from slepc4py import SLEPc

import ufl
import dolfinx

from tIGArx.LocalAssembly import assemble_matrix
from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots


def test_modal_analysis_1d():
    p = 3
    Nel = 100
    L = 1.0

    # Material constants for the Euler--Bernoulli beam problem:
    E = 1.0
    I = 1.0
    mu = 1.0

    n_modes = 5

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
    A.zeroRowsColumns(side_dofs, 1.0e15)

    M = assemble_matrix(dolfinx.fem.form(b), scalar_spline)
    # Mass of locked dofs is fixed to fix their ratio to the stiffness.
    M.zeroRowsColumns(side_dofs, 1.0)

    comm = spline.mesh.comm

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

    modes = np.zeros(n_modes, dtype=np.float64)

    # Look at the first N_MODES modes of the problem.
    for n in range(0, n_modes):
        u_vector_iga = A.getVecLeft()
        omega2 = eig_solver.getEigenpair(n, Vr=u_vector_iga)
        modes[n] = np.sqrt(omega2).real

    ref_modes = np.array(
        [
            3.516015157295631,
            22.034491615114383,
            61.697216040295345,
            120.90192836211132,
            199.85958583245662,
        ]
    )

    # It is somewhat interesting that there is such a "large" mismatch between
    # reference tigar and new local assembly. However, this order of error is i
    # in line with previous bi-harmonic tests, so it is likely due to the bad
    # conditioning of the problem.
    np.testing.assert_allclose(modes, ref_modes, rtol=3e-8)
