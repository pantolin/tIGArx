import numpy as np

import ufl
import dolfinx
from mpi4py import MPI

from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots
from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.common import mpirank
from tIGArx.solvers import solve_linear_variational_problem, \
    dolfinx_assemble_linear_variational_problem
from tIGArx.timing_util import perf_log


def biharmonic_local_2d():
    # Number of levels of refinement with which to run the Poisson problem.
    # (Note: Paraview output files will correspond to the last/highest level
    # of refinement.)
    N_LEVELS = 5

    # Array to store error at different refinement levels:
    L2_errors = np.zeros(N_LEVELS)
    energy_errors = np.zeros(N_LEVELS)

    # NOTE:  $L^2$ errors do not converge at optimal rate for the lowest feasible
    # spline degree (quadratic), as, to complete the Aubin--Nitsche duality
    # argument one needs the error to be more regular than this case allows.
    # Energy norm convergence remains optimal for all degrees and is measured
    # in this demo.  $L^2$ errors can be measured as well by un-commenting
    # several lines below.

    for level in range(0, N_LEVELS):

        ####### Preprocessing #######
        p = 4
        q = 4
        NELu = 8 * (2 ** level)
        NELv = 8 * (2 ** level)

        perf_log.start_timing("Dimension: " + str(NELu) + " x " + str(NELv), True)
        perf_log.start_timing("Generating control mesh")

        # Create a control mesh for which $\Omega = \widehat{\Omega}$.
        spline_mesh = ExplicitBSplineControlMesh(
            [p, q],
            [
                uniform_knots(p, -1.0, 1.0, NELu),
                uniform_knots(q, -1.0, 1.0, NELv)
            ]
        )

        if mpirank == 0:
            print("Setting up extracted spline...")

        perf_log.end_timing("Generating control mesh")
        perf_log.start_timing("Generating spline space")

        quad_order = 2 * max(p, q)
        spline = LocallyConstructedSpline.get_from_mesh_and_init(
            spline_mesh, quad_degree=quad_order, dofs_per_cp=1
        )

        perf_log.end_timing("Generating spline space")
        perf_log.start_timing("UFL problem setup")

        # Homogeneous coordinate representation of the trial function u.  Because
        # weights are 1 in the B-spline case, this can be used directly in the PDE,
        # without dividing through by weight.
        u = ufl.TrialFunction(spline.V)

        # Corresponding test function.
        v = ufl.TestFunction(spline.V)

        # Laplace operator, using spline's div and grad operations
        def lap(x):
            return spline.div(spline.grad(x))

        # Create a force, f, to manufacture the solution, soln
        x = spline.get_fe_cp_coordinates()
        soln = (ufl.cos(ufl.pi * x[0]) + 1.0) * (ufl.cos(ufl.pi * x[1]) + 1.0)
        f = lap(lap(soln))

        lhs = ufl.inner(lap(u), lap(v)) * spline.dx
        rhs = ufl.inner(f, v) * spline.dx
        u = dolfinx.fem.Function(spline.V)
        u.name = "u"

        perf_log.end_timing("UFL problem setup")
        perf_log.start_timing("Applying Dirichlet BCs")

        side_dofs = []
        scalar_spline = spline_mesh.getScalarSpline()
        for parametricDirection in [0, 1]:
            for side in [0, 1]:
                side_dofs.append(scalar_spline.getSideDofs(
                    parametricDirection,
                    side,
                    ##############################
                    layers=2,
                ))  # two layers of CPs
                ##############################

        side_dofs = np.array(np.unique(np.concatenate(side_dofs)), dtype=np.int32)
        dofs_values = np.zeros(len(side_dofs), dtype=np.float64)
        bcs = {"dirichlet": (side_dofs, dofs_values)}

        perf_log.end_timing("Applying Dirichlet BCs")

        cp_sol = solve_linear_variational_problem(
            lhs, rhs, scalar_spline, bcs, profile=True
        )

        perf_log.start_timing("Extracting solution")
        spline.extract_cp_solution_to_fe(cp_sol, u)
        perf_log.end_timing("Extracting solution")

        dolfinx_assemble_linear_variational_problem(lhs, rhs, profile=True)

        perf_log.end_timing("Dimension: " + str(NELu) + " x " + str(NELv), True)

        ####### Postprocessing #######

        # The solution, u, is in the homogeneous representation, but, again, for
        # B-splines with weight=1, this is the same as the physical representation.
        with dolfinx.io.VTXWriter(spline.mesh.comm, "results/u.bp", [u]) as vtx:
            vtx.write(0.0)

        # Compute and print the error in the discrete solution.
        L2_error_local = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(((u - soln) ** 2) * spline.dx)
        )
        comm = spline.comm
        L2_error = np.sqrt(comm.allreduce(L2_error_local, op=MPI.SUM))

        energy_error_local = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(lap(u - soln) ** 2 * spline.dx))
        energy_error = np.sqrt(comm.allreduce(energy_error_local, op=MPI.SUM))

        L2_errors[level] = L2_error
        energy_errors[level] = energy_error
        if level > 0:
            rate_L2 = np.log(L2_errors[level-1]/L2_errors[level])/np.log(2.0)
            rate_E = np.log(energy_errors[level - 1] /
                          energy_errors[level]) / np.log(2.0)
        else:
            rate_L2 = "--"
            rate_E = "--"
        if mpirank == 0:
            print("L2 Error for level "+str(level)+"     = "+str(L2_error)
                 +"  (rate = "+str(rate_L2)+")")
            print(
                "Energy error for level "
                + str(level)
                + " = "
                + str(energy_error)
                + "  (rate = "
                + str(rate_E)
                + ")"
            )


if __name__ == "__main__":
    biharmonic_local_2d()
