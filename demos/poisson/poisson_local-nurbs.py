import numpy as np
import dolfinx
import ufl

from mpi4py import MPI

from igakit.nurbs import NURBS as NURBS_ik

from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.common import mpirank
from tIGArx.NURBS import NURBSControlMesh

from tIGArx.solvers import solve_linear_variational_problem, \
    dolfinx_assemble_linear_variational_problem
from tIGArx.timing_util import perf_log


def local_poisson_nurbs_2d():
    N_LEVELS = 5

    # Array to store error at different refinement levels:
    L2_errors = np.zeros(N_LEVELS)

    for level in range(0, N_LEVELS):

        # Parameter determining level of refinement
        REF_LEVEL = level + 3
        n_new_knots = 2 ** REF_LEVEL

        perf_log.start_timing(
            "Dimension: " + str(n_new_knots) + " x " + str(n_new_knots), True
        )
        perf_log.start_timing("Generating igakit geometry")

        # Open knot vectors for a one-Bezier-element bi-unit square.
        u_knots = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        v_knots = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]

        # Array of control points, for a bi-unit square with the interior
        # parameterization distorted.
        cp_array = np.array(
            [
                [[-1.0, -1.0], [0.0, -1.0], [1.0, -1.0]],
                [[-1.0, 0.0], [0.7, 0.3], [1.0, 0.0]],
                [[-1.0, 1.0], [0.0, 1.0], [1.0, 1.0]],
            ]
        )

        # NOTE: Polynomial degree is determined based on the number of knots and
        # control points.  In this case, the NURBS is quadratic.

        # Create initial mesh
        ik_nurbs = NURBS_ik([u_knots, v_knots], cp_array)

        # Refinement

        h = 2.0 / float(n_new_knots)
        knot_list = []
        for i in range(0, n_new_knots - 1):
            knot_list += [
                float(i + 1) * h - 1.0,
            ]
        new_knots = np.array(knot_list)
        ik_nurbs.refine(0, new_knots)
        ik_nurbs.refine(1, new_knots)

        perf_log.end_timing("Generating igakit geometry")
        perf_log.start_timing("Generating mesh")

        spline_mesh = NURBSControlMesh(ik_nurbs)

        spline = LocallyConstructedSpline.get_from_mesh_and_init(
            spline_mesh, quad_degree=4, dofs_per_cp=1
        )

        spline.control_point_funcs[0].name = "FX"
        spline.control_point_funcs[1].name = "FY"
        spline.control_point_funcs[2].name = "FZ"
        spline.control_point_funcs[3].name = "FW"

        perf_log.end_timing("Generating mesh")
        perf_log.start_timing("UFL problem setup")

        # The trial function.  The function rationalize() creates a UFL Division
        # object which is the quotient of the homogeneous representation of the
        # function and the weight field from the control mesh.
        u = spline.rationalize(ufl.TrialFunction(spline.V))

        # Corresponding test function.
        v = spline.rationalize(ufl.TestFunction(spline.V))

        # Create a force, f, to manufacture the solution, soln
        x = spline.get_fe_coordinates()
        soln = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
        f = -spline.div(spline.grad(soln))

        # Set up and solve the Poisson problem
        a = ufl.inner(spline.grad(u), spline.grad(v)) * spline.dx
        L = ufl.inner(f, v) * spline.dx

        # FEniCS Function objects are always in the homogeneous representation; it
        # is a good idea to name variables in such a way as to recall this.
        u_hom = dolfinx.fem.Function(spline.V)
        u_hom.name = "u"

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
        bcs = {"dirichlet": (side_dofs, dofs_values)}

        perf_log.end_timing("Applying Dirichlet BCs")

        cp_sol = solve_linear_variational_problem(a, L, scalar_spline, bcs, profile=True)

        perf_log.start_timing("Extracting solution")
        spline.extract_cp_solution_to_fe(cp_sol, u_hom)
        perf_log.end_timing("Extracting solution")

        dolfinx_assemble_linear_variational_problem(a, L, profile=True)

        perf_log.end_timing(
            "Dimension: " + str(n_new_knots) + " x " + str(n_new_knots), True
        )

        ####### Postprocessing #######

        # The solution, u, is in the homogeneous representation, and
        # the geometry information
        with dolfinx.io.VTXWriter(spline.mesh.comm, "results/u.bp", [u_hom] + spline.control_point_funcs) as vtx:
            vtx.write(0.0)

        # Useful notes for plotting:
        #
        #  In Paraview, an appropriate vector field for the mesh
        #  warping and the weighted solution can be created using the Calculator
        #  filter.  E.g., in this case, the vector field to warp by would be
        #
        #   (FX/FW-coordsX)*iHat + (FY/FW-coordsY)*jHat + (FZ/FW-coordsZ)*kHat
        #
        #  in Paraview Calculator syntax, and the solution would be u/FW.

        # Compute and print the $L^2$ error in the discrete solution.

        L2_error_local = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(((spline.rationalize(u_hom) - soln) ** 2) * spline.dx))
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
    local_poisson_nurbs_2d()
