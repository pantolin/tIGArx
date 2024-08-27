import numpy as np
import dolfinx
import ufl

from mpi4py import MPI

from tIGArx.LocalAssembly import solve_linear_variational_problem, \
    dolfinx_assemble_linear_variational_problem
from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.common import mpirank
from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots

from tIGArx.ExtractedSpline import ExtractedSpline
from tIGArx.MultiFieldSplines import EqualOrderSpline
from tIGArx.timing_util import perf_log


def run_poisson():
    # Number of levels of refinement with which to run the Poisson problem.
    # (Note: Paraview output files correspond to the last level.)
    N_LEVELS = 4

    # Array to store error at different refinement levels:
    L2_errors = np.zeros(N_LEVELS)

    for level in range(0, N_LEVELS):

        ####### Preprocessing #######

        # Parameters determining the polynomial degree and number of elements in
        # each parametric direction.  By changing these and recording the error,
        # it is easy to see that the discrete solutions converge at optimal rates
        # under refinement.
        p = 3
        q = 3
        r = 3
        NELu = 4 * (2**level)
        NELv = 4 * (2**level)
        NELw = 4 * (2**level)

        # Parameters determining the position and size of the domain.
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0

        Lx = 1.0
        Ly = 1.0
        Lz = 1.0

        perf_log.start_timing("Dimension: " + str(NELu) + " x "
                              + str(NELv) + " x " + str(NELw), True)
        perf_log.start_timing("Generating control mesh")

        # Create a control mesh for which $\Omega = \widehat{\Omega}$.
        spline_mesh = ExplicitBSplineControlMesh(
            [p, q, r],
            [
                uniform_knots(p, x0, x0 + Lx, NELu),
                uniform_knots(q, y0, y0 + Ly, NELv),
                uniform_knots(r, z0, z0 + Lz, NELw)
            ]
        )

        perf_log.end_timing("Generating control mesh")
        perf_log.start_timing("Generating spline space")

        quad_order = 3 * max(p, q, r)
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

        # Create a force, f, to manufacture the solution, soln
        x = spline.get_fe_cp_coordinates()
        soln = ufl.sin(ufl.pi * (x[0] - x0) / Lx) * \
            ufl.sin(ufl.pi * (x[1] - y0) / Ly) * \
            ufl.sin(ufl.pi * (x[2] - z0) / Lz)
        f = -spline.div(spline.grad(soln))

        # Set up and solve the Poisson problem
        a = ufl.inner(spline.grad(u), spline.grad(v)) * spline.dx
        L = ufl.inner(f, v) * spline.dx
        u = dolfinx.fem.Function(spline.V)
        u.name = "u"

        perf_log.end_timing("UFL problem setup")
        perf_log.start_timing("Applying Dirichlet BCs")

        side_dofs = []
        scalar_spline = spline_mesh.getScalarSpline()
        for parametricDirection in [0, 1, 2]:
            for side in [0, 1]:
                side_dofs.append(scalar_spline.getSideDofs(parametricDirection, side))

        # Filter for unique dofs
        side_dofs = np.array(np.unique(np.concatenate(side_dofs)), dtype=np.int32)
        dofs_values = np.zeros(len(side_dofs), dtype=np.float64)

        bcs = {"dirichlet": (side_dofs, dofs_values)}

        perf_log.end_timing("Applying Dirichlet BCs")

        cp_sol = solve_linear_variational_problem(a, L, scalar_spline, bcs, profile=True)

        perf_log.start_timing("Extracting solution")

        # Extract the solution at the control points
        spline.extract_cp_solution_to_fe(cp_sol, u)

        perf_log.end_timing("Extracting solution")

        dolfinx_assemble_linear_variational_problem(a, L, profile=True)

        perf_log.end_timing("Dimension: " + str(NELu) + " x "
                            + str(NELv) + " x " + str(NELw), True)

        ####### Postprocessing #######

        # Compute and print the $L^2$ error in the discrete solution.
        L2_error_local = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(((u - soln) ** 2) * spline.dx))
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
    run_poisson()
