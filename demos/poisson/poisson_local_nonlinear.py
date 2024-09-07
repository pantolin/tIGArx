import numpy as np
import dolfinx
import ufl

from mpi4py import MPI

from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.common import mpirank
from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots

from tIGArx.timing_util import perf_log


def local_poisson():
    # Number of levels of refinement with which to run the Poisson problem.
    # (Note: Paraview output files correspond to the last level.)
    N_LEVELS = 5

    # Array to store error at different refinement levels
    L2_errors = np.zeros(N_LEVELS)

    for level in range(0, N_LEVELS):

        p = 4
        q = 4
        NELu = 8 * (2**level)
        NELv = 8 * (2**level)

        x0 = 0.0
        y0 = 0.0
        Lx = 1.0
        Ly = 1.0

        alpha = 10.0

        perf_log.start_timing("Dimension: " + str(NELu) + " x " + str(NELv), True)
        perf_log.start_timing("Generating control mesh")

        # Create a control mesh for which $\Omega = \widehat{\Omega}$.
        spline_mesh = ExplicitBSplineControlMesh(
            [p, q],
            [
                uniform_knots(p, x0, x0 + Lx, NELu),
                uniform_knots(q, y0, y0 + Ly, NELv)
            ]
        )

        perf_log.end_timing("Generating control mesh")
        perf_log.start_timing("Generating spline space")

        quad_order = 2 * max(p, q)
        spline = LocallyConstructedSpline.get_from_mesh_and_init(
            spline_mesh, quad_degree=quad_order, dofs_per_cp=1
        )

        perf_log.end_timing("Generating spline space")
        perf_log.start_timing("UFL problem setup")

        # Create a force, f, to manufacture the solution, soln
        x = spline.get_fe_coordinates()
        soln = ufl.sin(ufl.pi * (x[0] - x0) / Lx) * \
               ufl.sin(ufl.pi * (x[1] - y0) / Ly)
        f = -spline.div(spline.grad(soln)) + alpha * soln * soln * soln

        # u = spline.project(soln, rationalize=False, lump_mass=False)
        u = dolfinx.fem.Function(spline.V)
        u.name = "u"

        v = ufl.TestFunction(spline.V)

        # Define the bilinear form and linear form of the PDE
        residual = (ufl.inner(spline.grad(u), spline.grad(v))
                    + alpha * ufl.inner(u, u) * ufl.inner(u, v)
                    - ufl.inner(f, v)) * spline.dx
        jacobian = ufl.derivative(residual, u)


        perf_log.end_timing("UFL problem setup")
        perf_log.start_timing("Applying Dirichlet BCs")

        side_dofs = []
        scalar_spline = spline_mesh.getScalarSpline()
        for parametricDirection in [0, 1]:
            for side in [0, 1]:
                side_dofs.append(scalar_spline.getSideDofs(parametricDirection, side))

        # Filter for unique dofs
        side_dofs = np.array(np.unique(np.concatenate(side_dofs)), dtype=np.int32)
        dofs_values = np.zeros(len(side_dofs), dtype=np.float64)

        # set all dofs in u which are not on the side to 0
        # u.x.array[np.setdiff1d(np.arange(len(u.x.array)), side_dofs)] = 0.0

        bcs = {"dirichlet": (side_dofs, dofs_values)}

        perf_log.end_timing("Applying Dirichlet BCs")

        spline.solve_nonlinear_variational_problem(jacobian, residual, u, bcs)

        perf_log.end_timing("Dimension: " + str(NELu) + " x " + str(NELv))

        ####### Postprocessing #######

        # The solution, u, is in the homogeneous representation, but, again, for
        # B-splines with weight=1, this is the same as the physical representation.
        with dolfinx.io.VTXWriter(spline.mesh.comm,
                                  "results/u.bp",
                                  [u]) as vtx:
            vtx.write(0.0)

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
    local_poisson()
