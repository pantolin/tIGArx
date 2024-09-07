import numpy as np

import dolfinx
import ufl

from mpi4py import MPI

from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots
from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.solvers import solve_linear_variational_problem


def test_biharmonic_2d():
    p = 4
    q = 5

    NELu = 21
    NELv = 16

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q],
        [
            uniform_knots(p, -1.0, 1.0, NELu),
            uniform_knots(q, -1.0, 1.0, NELv)
        ]
    )

    quad_order = 2 * max(p, q)
    spline = LocallyConstructedSpline.get_from_mesh_and_init(
        spline_mesh, quad_degree=quad_order, dofs_per_cp=1
    )

    u = ufl.TrialFunction(spline.V)
    v = ufl.TestFunction(spline.V)

    # Laplace operator, using spline's div and grad operations
    def lap(x):
        return spline.div(spline.grad(x))

    x = spline.get_fe_coordinates()
    soln = (ufl.cos(ufl.pi * x[0]) + 1.0) * (ufl.cos(ufl.pi * x[1]) + 1.0)
    f = lap(lap(soln))

    lhs = ufl.inner(lap(u), lap(v)) * spline.dx
    rhs = ufl.inner(f, v) * spline.dx
    u = dolfinx.fem.Function(spline.V)
    u.name = "u"

    side_dofs = []
    scalar_spline = spline_mesh.getScalarSpline()
    for parametricDirection in [0, 1]:
        for side in [0, 1]:
            side_dofs.append(scalar_spline.getSideDofs(
                parametricDirection,
                side,
                layers=2,
            ))

    side_dofs = np.array(np.unique(np.concatenate(side_dofs)), dtype=np.int32)
    dofs_values = np.zeros(len(side_dofs), dtype=np.float64)
    bcs = {"dirichlet": (side_dofs, dofs_values)}

    cp_sol = solve_linear_variational_problem(lhs, rhs, scalar_spline, bcs)
    spline.extract_cp_solution_to_fe(cp_sol, u)

    L2_error_local = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(((u - soln) ** 2) * spline.dx)
    )
    comm = spline.comm
    L2_error = np.sqrt(comm.allreduce(L2_error_local, op=MPI.SUM))

    energy_error_local = dolfinx.fem.assemble_scalar(
        dolfinx.fem.form(lap(u - soln) ** 2 * spline.dx))
    energy_error = np.sqrt(comm.allreduce(energy_error_local, op=MPI.SUM))

    # print("L2 error: ", L2_error)
    # print("Energy error: ", energy_error)

    assert np.isclose(L2_error, 0.0, atol=1e-6)
    assert np.isclose(energy_error, 0.0, atol=3e-3)
