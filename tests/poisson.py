import numpy as np

import ufl
import dolfinx

from mpi4py import MPI

from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots
from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.solvers import solve_linear_variational_problem


def test_bspline_poisson_2d():
    p = 3
    q = 4

    NELu = 21
    NELv = 16

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
        spline_mesh, quad_degree=quad_order, dofs_per_cp=1
    )

    u = ufl.TrialFunction(spline.V)
    v = ufl.TestFunction(spline.V)

    x = spline.get_fe_cp_coordinates()
    soln = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = -spline.div(spline.grad(soln))

    a = ufl.inner(spline.grad(u), spline.grad(v)) * spline.dx
    L = ufl.inner(f, v) * spline.dx
    u = dolfinx.fem.Function(spline.V)
    u.name = "u"

    side_dofs = []
    scalar_spline = spline_mesh.getScalarSpline()
    for parametricDirection in [0, 1]:
        for side in [0, 1]:
            side_dofs.append(scalar_spline.getSideDofs(parametricDirection, side))

    # Filter for unique dofs
    side_dofs = np.array(np.unique(np.concatenate(side_dofs)), dtype=np.int32)
    dofs_values = np.zeros(len(side_dofs), dtype=np.float64)
    bcs = {"dirichlet": (side_dofs, dofs_values)}

    cp_sol = solve_linear_variational_problem(a, L, scalar_spline, bcs, profile=False)
    spline.extract_cp_solution_to_fe(cp_sol, u)

    L2_error_local = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(((u - soln) ** 2) * spline.dx))
    comm = spline.comm
    L2_error = np.sqrt(comm.allreduce(L2_error_local, op=MPI.SUM))

    assert np.isclose(L2_error, 0.0, atol=3e-7)


def test_bspline_poisson_3d():
    p = 2
    q = 3
    r = 4

    NELu = 14
    NELv = 9
    NELw = 7

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
        spline_mesh, quad_degree=quad_order, dofs_per_cp=1
    )

    u = ufl.TrialFunction(spline.V)
    v = ufl.TestFunction(spline.V)

    x = spline.get_fe_cp_coordinates()
    soln = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1]) * ufl.sin(ufl.pi * x[2])
    f = -spline.div(spline.grad(soln))

    a = ufl.inner(spline.grad(u), spline.grad(v)) * spline.dx
    L = ufl.inner(f, v) * spline.dx
    u = dolfinx.fem.Function(spline.V)
    u.name = "u"

    side_dofs = []
    scalar_spline = spline_mesh.getScalarSpline()
    for parametricDirection in [0, 1, 2]:
        for side in [0, 1]:
            side_dofs.append(scalar_spline.getSideDofs(parametricDirection, side))

    # Filter for unique dofs
    side_dofs = np.array(np.unique(np.concatenate(side_dofs)), dtype=np.int32)
    dofs_values = np.zeros(len(side_dofs), dtype=np.float64)
    bcs = {"dirichlet": (side_dofs, dofs_values)}

    cp_sol = solve_linear_variational_problem(a, L, scalar_spline, bcs, profile=False)
    spline.extract_cp_solution_to_fe(cp_sol, u)

    L2_error_local = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(((u - soln) ** 2) * spline.dx))
    comm = spline.comm
    L2_error = np.sqrt(comm.allreduce(L2_error_local, op=MPI.SUM))

    assert np.isclose(L2_error, 0.0, atol=3e-5)


def test_bspline_poisson_2d_nonlinear():
    p = 3
    q = 4

    NELu = 21
    NELv = 16

    alpha = 10.0

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
        spline_mesh, quad_degree=quad_order, dofs_per_cp=1
    )

    x = spline.get_fe_cp_coordinates()
    soln = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = -spline.div(spline.grad(soln)) + alpha * soln * soln * soln

    u = dolfinx.fem.Function(spline.V)
    u.name = "u"
    v = ufl.TestFunction(spline.V)

    residual = (ufl.inner(spline.grad(u), spline.grad(v))
                + alpha * ufl.inner(u, u) * ufl.inner(u, v)
                - ufl.inner(f, v)) * spline.dx
    jacobian = ufl.derivative(residual, u)

    side_dofs = []
    scalar_spline = spline_mesh.getScalarSpline()
    for parametricDirection in [0, 1]:
        for side in [0, 1]:
            side_dofs.append(scalar_spline.getSideDofs(parametricDirection, side))

    # Filter for unique dofs
    side_dofs = np.array(np.unique(np.concatenate(side_dofs)), dtype=np.int32)
    dofs_values = np.zeros(len(side_dofs), dtype=np.float64)
    bcs = {"dirichlet": (side_dofs, dofs_values)}

    spline.solve_nonlinear_variational_problem(jacobian, residual, u, bcs)

    L2_error_local = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(((u - soln) ** 2) * spline.dx))
    comm = spline.comm
    L2_error = np.sqrt(comm.allreduce(L2_error_local, op=MPI.SUM))

    assert np.isclose(L2_error, 0.0, atol=3e-7)
