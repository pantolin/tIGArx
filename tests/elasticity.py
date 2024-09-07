import numpy as np

import ufl
import dolfinx

from mpi4py import MPI

from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots
from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.solvers import solve_linear_variational_problem
from tIGArx.utils import interleave_and_expand


def test_bspline_elasticity_2d():
    p = 3
    q = 4

    NELu = 21
    NELv = 16

    # Material parameters
    E = 1000.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Create a control mesh for which $\Omega = \widehat{\Omega}$.
    spline_mesh = ExplicitBSplineControlMesh(
        [p, q],
        [
            uniform_knots(p, 0.0, 1.0, NELu),
            uniform_knots(q, 0.0, 1.0, NELv)
        ]
    )

    quad_order = 2 * max(p, q)
    spline = LocallyConstructedSpline.get_from_mesh_and_init(
        spline_mesh, quad_degree=quad_order, dofs_per_cp=2
    )

    u = ufl.TrialFunction(spline.V)
    v = ufl.TestFunction(spline.V)

    def epsilon(u):
        return ufl.sym(
            spline.grad(u)
        )  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lmbda * spline.div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    x = spline.get_fe_coordinates()
    soln = ufl.as_vector([ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]),
                          ufl.sin(2.0 * ufl.pi * x[0]) * ufl.sin(2.0 * ufl.pi * x[1])])
    f = -spline.div(sigma(soln))

    a = ufl.inner(sigma(u), epsilon(v)) * spline.dx
    L = ufl.dot(f, v) * spline.dx
    u = dolfinx.fem.Function(spline.V)
    u.name = "u"

    side_dofs = []
    scalar_spline = spline_mesh.getScalarSpline()
    for parametricDirection in [0, 1]:
        for side in [0, 1]:
            side_dofs.append(scalar_spline.getSideDofs(parametricDirection, side))

    # Filter for unique dofs
    side_dofs = np.array(np.unique(np.concatenate(side_dofs)), dtype=np.int32)
    side_dofs = interleave_and_expand(side_dofs, 2)
    side_dofs = np.array(side_dofs, dtype=np.int32)

    dofs_values = np.zeros(len(side_dofs), dtype=np.float64)
    bcs = {"dirichlet": (side_dofs, dofs_values)}

    cp_sol = solve_linear_variational_problem(a, L, scalar_spline, bcs, profile=False)
    spline.extract_cp_solution_to_fe(cp_sol, u)

    L2_error_local = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(((u - soln) ** 2) * spline.dx))
    comm = spline.comm
    L2_error = np.sqrt(comm.allreduce(L2_error_local, op=MPI.SUM))

    print(L2_error)

    assert np.isclose(L2_error, 0.0, atol=5e-6)


def test_bspline_elasticity_3d():
    p = 1
    q = 2
    r = 3

    NELu = 21
    NELv = 10
    NELw = 7

    # Material parameters
    E = 1000.0
    nu = 0.3
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # Create a control mesh for which $\Omega = \widehat{\Omega}$.
    spline_mesh = ExplicitBSplineControlMesh(
        [p, q, r],
        [
            uniform_knots(p, 0.0, 1.0, NELu),
            uniform_knots(q, 0.0, 1.0, NELv),
            uniform_knots(r, 0.0, 1.0, NELw)
        ]
    )

    quad_order = 2 * max(p, q, r)
    spline = LocallyConstructedSpline.get_from_mesh_and_init(
        spline_mesh, quad_degree=quad_order, dofs_per_cp=3
    )

    u = ufl.TrialFunction(spline.V)
    v = ufl.TestFunction(spline.V)

    def epsilon(u):
        return ufl.sym(
            spline.grad(u)
        )  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return lmbda * spline.div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    x = spline.get_fe_coordinates()
    soln0 = (ufl.sin(ufl.pi * x[0])
             * ufl.sin(ufl.pi * x[1])
             * ufl.sin(ufl.pi * x[2]))
    soln1 = (ufl.sin(2.0 * ufl.pi * x[0])
             * ufl.sin(2.0 * ufl.pi * x[1])
             * ufl.sin(2.0 * ufl.pi * x[2]))
    soln2 = (ufl.sin(2.0 * ufl.pi * x[0])
             * ufl.sin(2.0 * ufl.pi * x[1])
             * ufl.sin(2.0 * ufl.pi * x[2]))

    soln = ufl.as_vector([soln0, soln1, soln2])
    f = -spline.div(sigma(soln))

    a = ufl.inner(sigma(u), epsilon(v)) * spline.dx
    L = ufl.inner(f, v) * spline.dx
    u = dolfinx.fem.Function(spline.V)
    u.name = "u"

    side_dofs = []
    scalar_spline = spline_mesh.getScalarSpline()
    for parametricDirection in [0, 1, 2]:
        for side in [0, 1]:
            side_dofs.append(scalar_spline.getSideDofs(parametricDirection, side))

    side_dofs = np.array(np.unique(np.concatenate(side_dofs)), dtype=np.int32)
    side_dofs = interleave_and_expand(side_dofs, 3)
    side_dofs = np.array(side_dofs, dtype=np.int32)

    dofs_values = np.zeros(len(side_dofs), dtype=np.float64)
    bcs = {"dirichlet": (side_dofs, dofs_values)}

    cp_sol = solve_linear_variational_problem(a, L, scalar_spline, bcs, profile=False)
    spline.extract_cp_solution_to_fe(cp_sol, u)

    L2_error_local = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(((u - soln) ** 2) * spline.dx))
    comm = spline.comm
    L2_error = np.sqrt(comm.allreduce(L2_error_local, op=MPI.SUM))
    print(L2_error)

    assert np.isclose(L2_error, 0.0, atol=3e-3)