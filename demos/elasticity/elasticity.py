"""
The "hello, world" of computational PDEs:  Solve the Elaticity equation, 
verifying accuracy via the method of manufactured solutions.  

This example uses the simplest IGA discretization, namely, explicit B-splines
in which parametric and physical space are the same.
"""

import numpy as np
import dolfinx
import ufl

from mpi4py import MPI

from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.common import mpirank
from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots
from tIGArx.solvers import solve_linear_variational_problem, \
    dolfinx_assemble_linear_variational_problem

from tIGArx.timing_util import perf_log
from tIGArx.utils import interleave_and_expand


def run_elasticity():

    # Number of levels of refinement with which to run the Elasticity problem.
    # (Note: Paraview output files will correspond to the last/highest level
    # of refinement.)
    N_LEVELS = 5

    # Array to store error at different refinement levels:
    L2_errors = np.zeros(N_LEVELS)

    for level in range(0, N_LEVELS):

        p = 4
        q = 4
        NELu = 8 * (2**level)
        NELv = 8 * (2**level)

        # Material parameters
        E = 1000.0
        nu = 0.3
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Parameters determining the position and size of the domain.
        x0 = 0.0
        y0 = 0.0
        Lx = 1.0
        Ly = 1.0

        perf_log.start_timing("Dimension: " + str(NELu) + " x " + str(NELv), True)
        perf_log.start_timing("Generating control mesh")

        # Create a control mesh for which $\Omega = \widehat{\Omega}$.
        spline_mesh = ExplicitBSplineControlMesh(
            [p, q],
            [uniform_knots(p, x0, x0 + Lx, NELu), uniform_knots(q, y0, y0 + Ly, NELv)],
        )

        perf_log.end_timing("Generating control mesh")
        perf_log.start_timing("Generating spline space")

        quad_order = 2 * max(p, q)
        spline = LocallyConstructedSpline.get_from_mesh_and_init(
            spline_mesh, quad_degree=quad_order, dofs_per_cp=2
        )

        perf_log.end_timing("Generating spline space")
        perf_log.start_timing("UFL problem setup")

        # Homogeneous coordinate representation of the trial function u.  Because
        # weights are 1 in the B-spline case, this can be used directly in the PDE,
        # without dividing through by weight.
        u = ufl.TrialFunction(spline.V)

        # Corresponding test function.
        v = ufl.TestFunction(spline.V)

        # Set up elasticity operators.
        def epsilon(u):
            return ufl.sym(
                spline.grad(u)
            )  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

        def sigma(u):
            return lmbda * spline.div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

        # Create a force, f, to manufacture the solution, soln
        x = spline.get_fe_cp_coordinates()
        soln0 = ufl.sin(ufl.pi * (x[0] - x0) / Lx) * ufl.sin(ufl.pi * (x[1] - y0) / Ly)
        soln1 = (ufl.sin(2.0 * ufl.pi * (x[0] - x0) / Lx)
                 * ufl.sin(2.0 * ufl.pi * (x[1] - y0) / Ly)
                 )
        soln = ufl.as_vector([soln0, soln1])
        f = -spline.div(sigma(soln))

        # Set up and solve the Elasticity problem
        a = ufl.inner(sigma(u), epsilon(v)) * spline.dx
        L = ufl.dot(f, v) * spline.dx
        u = dolfinx.fem.Function(spline.V)
        u.name = "u"

        perf_log.end_timing("UFL problem setup")
        perf_log.start_timing("Applying Dirichlet BCs")

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

        perf_log.end_timing("Applying Dirichlet BCs")

        cp_sol = solve_linear_variational_problem(a, L, scalar_spline, bcs, profile=True)

        perf_log.start_timing("Extracting solution")
        spline.extract_cp_solution_to_fe(cp_sol, u)
        perf_log.end_timing("Extracting solution")

        dolfinx_assemble_linear_variational_problem(a, L, profile=True)

        perf_log.end_timing("Dimension: " + str(NELu) + " x " + str(NELv))

        ####### Postprocessing #######

        # The solution, u, is in the homogeneous representation, but, again, for
        # B-splines with weight=1, this is the same as the physical representation.
        with dolfinx.io.VTXWriter(spline.mesh.comm, "results/u.bp", [u]) as vtx:
            vtx.write(0.0)

        # Compute and print the $L^2$ error in the discrete solution.
        L2_error_local = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(ufl.inner(u - soln, u - soln) * spline.dx)
        )
        comm = spline.comm
        L2_error = np.sqrt(comm.allreduce(L2_error_local, op=MPI.SUM))

        L2_errors[level] = L2_error
        if level > 0:
            rate = np.log(L2_errors[level - 1] / L2_errors[level]) / np.log(2.0)
        else:
            rate = "--"
        if mpirank == 0:
            print(
                "L2 Error for level "
                + str(level)
                + " = "
                + str(L2_error)
                + "  (rate = "
                + str(rate)
                + ")"
            )


if __name__ == "__main__":
    run_elasticity()
