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

from tIGArx.LocalAssembly import solve_linear_variational_problem, \
    dolfinx_assemble_linear_variational_problem
from tIGArx.common import mpirank
from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots

from tIGArx.ExtractedSpline import ExtractedSpline
from tIGArx.MultiFieldSplines import EqualOrderSpline
from tIGArx.timing_util import perf_log
from tIGArx.utils import stack_and_shift


def run_elasticity():

    # Number of levels of refinement with which to run the Elasticity problem.
    # (Note: Paraview output files will correspond to the last/highest level
    # of refinement.)
    N_LEVELS = 4

    # Array to store error at different refinement levels:
    L2_errors = np.zeros(N_LEVELS)

    profile = True

    for level in range(0, N_LEVELS):

        ####### Preprocessing #######

        # Parameters determining the polynomial degree and number of elements in
        # each parametric direction.  By changing these and recording the error,
        # it is easy to see that the discrete solutions converge at optimal rates
        # under refinement.
        p = 1
        q = 1
        r = 1
        NELu = 4 * (2**level)
        NELv = 4 * (2**level)
        NELw = 4 * (2**level)

        # Material parameters
        E = 1000.0
        nu = 0.3
        mu = E / (2.0 * (1.0 + nu))
        lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # Parameters determining the position and size of the domain.
        x0 = 0.0
        y0 = 0.0
        z0 = 0.0

        Lx = 1.0
        Ly = 1.0
        Lz = 1.0

        perf_log.start_timing("Dimension: " + str(NELu) + " x " + str(NELv), True)
        perf_log.start_timing("Generating control mesh")

        # Create a control mesh for which $\Omega = \widehat{\Omega}$.
        splineMesh = ExplicitBSplineControlMesh(
            [p, q, r],
            [
                uniform_knots(p, x0, x0 + Lx, NELu),
                uniform_knots(q, y0, y0 + Ly, NELv),
                uniform_knots(r, z0, z0 + Lz, NELw)
            ],
        )

        perf_log.end_timing("Generating control mesh")
        perf_log.start_timing("Generating spline generator")

        # Create a spline generator for a spline with a vector field on the
        # given control mesh, where the field is the same as the one used
        # to determine the mapping $\mathbf{F}:\widehat{\Omega}\to\Omega$.
        splineGenerator = EqualOrderSpline(2, splineMesh)

        perf_log.end_timing("Generating spline generator")
        perf_log.start_timing("Setting Dirichlet bcs")

        # Set Dirichlet boundary conditions for both fields, on both
        # ends of the domain, in both directions.
        for field in range(3):
            scalarSpline = splineGenerator.getScalarSpline(field)
            for parametricDirection in [0, 1, 2]:
                for side in [0, 1]:
                    sideDofs = scalarSpline.getSideDofs(parametricDirection, side)
                    splineGenerator.addZeroDofs(field, sideDofs)

        perf_log.end_timing("Setting Dirichlet bcs")
        perf_log.start_timing("Setting up extracted spline")

        # Alternative: set BCs based on location of corresponding control points.
        # (Note that this only makes sense for splineGenerator of type
        # EqualOrderSpline; for non-equal-order splines, there is not
        # a one-to-one correspondence between degrees of freedom and geometry
        # control points.)

        # field = 0
        # def get_boundary(x, on_boundary):
        #     return (near(x[0],x0) or near(x[0],x0+Lx)
        #             or near(x[1],y0) or near(x[1],y0+Ly))
        # splineGenerator.addZeroDofsByLocation(get_boundary(),field)

        # Write extraction data to the filesystem.
        DIR = "./extraction"
        # FIXME to uncomment.
        # splineGenerator.writeExtraction(DIR)

        ####### Analysis #######

        # Choose the quadrature degree to be used throughout the analysis.
        # In IGA, especially with rational spline spaces, under-integration is a
        # fact of life, but this does not impair optimal convergence.
        QUAD_DEG = 3 * max(p, q, r)

        # Create the extracted spline directly from the generator.
        # As of version 2019.1, this is required for using quad/hex elements in
        # parallel.
        spline = ExtractedSpline(splineGenerator, QUAD_DEG)
        # spline = ExtractedSpline(DIR, QUAD_DEG)

        # Alternative: Can read the extracted spline back in from the filesystem.
        # For quad/hex elements, in version 2019.1, this only works in serial.

        # spline = ExtractedSpline(DIR,QUAD_DEG)

        perf_log.end_timing("Setting up extracted spline")
        perf_log.start_timing("Solving", True)
        perf_log.start_timing("Setting up problem")

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
        x = spline.spatialCoordinates()
        soln0 = (ufl.sin(ufl.pi * (x[0] - x0) / Lx)
                 * ufl.sin(ufl.pi * (x[1] - y0) / Ly)
                 * ufl.sin(ufl.pi * (x[2] - z0) / Lz))

        soln1 = (ufl.sin(2.0 * ufl.pi * (x[0] - x0) / Lx)
                 * ufl.sin(2.0 * ufl.pi * (x[1] - y0) / Ly)
                 * ufl.sin(2.0 * ufl.pi * (x[2] - z0) / Lz))

        soln2 = (ufl.sin(3.0 * ufl.pi * (x[0] - x0) / Lx)
                 * ufl.sin(3.0 * ufl.pi * (x[1] - y0) / Ly)
                 * ufl.sin(3.0 * ufl.pi * (x[2] - z0) / Lz))

        soln = ufl.as_vector([soln0, soln1, soln2])
        f = -spline.div(sigma(soln))

        # Set up and solve the Elasticity problem
        a = ufl.inner(sigma(u), epsilon(v)) * spline.dx
        L = ufl.dot(f, v) * spline.dx
        u = dolfinx.fem.Function(spline.V)
        u.name = "u"

        perf_log.end_timing("Setting up problem")
        perf_log.start_timing("Solve problem")

        # spline.solveLinearVariationalProblem(a == L, u)

        side_dofs = []
        for parametricDirection in [0, 1, 2]:
            for side in [0, 1]:
                side_dofs.append(scalarSpline.getSideDofs(parametricDirection, side))

        # Filter for unique dofs
        side_dofs = np.array(np.unique(np.concatenate(side_dofs)), dtype=np.int32)
        side_dofs = stack_and_shift(side_dofs, 3, scalarSpline.getNcp())
        side_dofs = np.array(side_dofs, dtype=np.int32)
        dofs_values = np.zeros(len(side_dofs), dtype=np.float64)

        bcs = {"dirichlet": (side_dofs, dofs_values)}
        cp_sol = solve_linear_variational_problem(a, L, scalarSpline, bcs, profile=True)

        # Using the global matrix because it is available, the same
        # effect could be achieved by evaluating the splines at the
        # desired points and multiplying by the control point
        # contribution to the solution.
        sol = splineGenerator.M * cp_sol
        size = u.x.array.size
        u.x.array[:size] = sol.array_r

        spline.solveLinearVariationalProblem(a == L, u)

        perf_log.end_timing("Solve problem")
        perf_log.end_timing("Solving")
        perf_log.end_timing("Dimension: " + str(NELu) + " x " + str(NELv))

        dolfinx_assemble_linear_variational_problem(a, L, profile=True)

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
