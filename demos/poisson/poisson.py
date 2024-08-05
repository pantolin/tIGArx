"""
The "hello, world" of computational PDEs:  Solve the Poisson equation, 
verifying accuracy via the method of manufactured solutions.  

This example uses the simplest IGA discretization, namely, explicit B-splines
in which parametric and physical space are the same.
"""
import time

from tIGArx.common import mpirank, EqualOrderSpline, ExtractedSpline
from tIGArx.BSplines import ExplicitBSplineControlMesh, uniformKnots

import dolfinx
import ufl

from mpi4py import MPI
import numpy as np

# Number of levels of refinement with which to run the Poisson problem.
# (Note: Paraview output files will correspond to the last/highest level
# of refinement.)
N_LEVELS = 3

# Array to store error at different refinement levels:
L2_errors = np.zeros(N_LEVELS)

profile = True

for level in range(0, N_LEVELS):

    ####### Preprocessing #######

    # Parameters determining the polynomial degree and number of elements in
    # each parametric direction.  By changing these and recording the error,
    # it is easy to see that the discrete solutions converge at optimal rates
    # under refinement.
    p = 3
    q = 3
    NELu = 10 * (2**level)
    NELv = 10 * (2**level)

    # Parameters determining the position and size of the domain.
    x0 = 0.0
    y0 = 0.0
    Lx = 1.0
    Ly = 1.0

    if mpirank == 0:
        print("Dimension: ", NELu, "x", NELv)
        print("Generating extraction...")
        start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    # Create a control mesh for which $\Omega = \widehat{\Omega}$.
    splineMesh = ExplicitBSplineControlMesh(
        [p, q], [uniformKnots(p, x0, x0 + Lx, NELu),
                 uniformKnots(q, y0, y0 + Ly, NELv)]
    )

    if mpirank == 0:
        if profile:
            print("Generating B-spline control mesh: ",
                  (time.clock_gettime_ns(time.CLOCK_MONOTONIC) - start) / 1e9)
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    # Create a spline generator for a spline with a single scalar field on the
    # given control mesh, where the scalar field is the same as the one used
    # to determine the mapping $\mathbf{F}:\widehat{\Omega}\to\Omega$.
    # FIXME
    splineGenerator = EqualOrderSpline(1, splineMesh)
    # splineGenerator = EqualOrderSpline(2, splineMesh)

    if mpirank == 0:
        if profile:
            print("Generating spline generator: ",
                  (time.clock_gettime_ns(time.CLOCK_MONOTONIC) - start) / 1e9)
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    # Set Dirichlet boundary conditions on the 0-th (and only) field, on both
    # ends of the domain, in both directions.
    field = 0
    scalarSpline = splineGenerator.getScalarSpline(field)
    for parametricDirection in [0, 1]:
        for side in [0, 1]:
            sideDofs = scalarSpline.getSideDofs(parametricDirection, side)
            splineGenerator.addZeroDofs(field, sideDofs)

    if mpirank == 0:
        if profile:
            print("Setting Dirichlet bcs: ",
                  (time.clock_gettime_ns(time.CLOCK_MONOTONIC) - start) / 1e9)
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

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

    if mpirank == 0:
        print("Setting up extracted spline...")

    # Choose the quadrature degree to be used throughout the analysis.
    # In IGA, especially with rational spline spaces, under-integration is a
    # fact of life, but this does not impair optimal convergence.
    QUAD_DEG = 2 * max(p, q)

    # Create the extracted spline directly from the generator.
    # As of version 2019.1, this is required for using quad/hex elements in
    # parallel.
    spline = ExtractedSpline(splineGenerator, QUAD_DEG)
    # spline = ExtractedSpline(DIR, QUAD_DEG)

    # Alternative: Can read the extracted spline back in from the filesystem.
    # For quad/hex elements, in version 2019.1, this only works in serial.

    # spline = ExtractedSpline(DIR,QUAD_DEG)

    if mpirank == 0:
        if profile:
            print("Setting up extracted spline: ",
                  (time.clock_gettime_ns(time.CLOCK_MONOTONIC) - start) / 1e9)
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        print("Solving...")

    # Homogeneous coordinate representation of the trial function u.  Because
    # weights are 1 in the B-spline case, this can be used directly in the PDE,
    # without dividing through by weight.
    u = ufl.TrialFunction(spline.V)

    # Corresponding test function.
    v = ufl.TestFunction(spline.V)

    # Create a force, f, to manufacture the solution, soln
    x = spline.spatialCoordinates()
    soln = ufl.sin(ufl.pi * (x[0] - x0) / Lx) * \
        ufl.sin(ufl.pi * (x[1] - y0) / Ly)
    f = -spline.div(spline.grad(soln))

    # Set up and solve the Poisson problem
    a = ufl.inner(spline.grad(u), spline.grad(v)) * spline.dx
    L = ufl.inner(f, v) * spline.dx
    u = dolfinx.fem.Function(spline.V)
    u.name = "u"

    if mpirank == 0:
        if profile:
            print("Setting up problem: ", (time.clock_gettime_ns(time.CLOCK_MONOTONIC) - start) / 1e9)
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    spline.solveLinearVariationalProblem(a == L, u)

    if mpirank == 0:
        if profile:
            print("Solving problem: ", (time.clock_gettime_ns(time.CLOCK_MONOTONIC) - start) / 1e9)
            start = time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    ####### Postprocessing #######

    # The solution, u, is in the homogeneous representation, but, again, for
    # B-splines with weight=1, this is the same as the physical representation.
    with dolfinx.io.VTXWriter(spline.mesh.comm, "results/u.bp", [u]) as vtx:
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
