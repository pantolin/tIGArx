import os.path

import numpy as np
import dolfinx
import ufl

from dolfinx import default_real_type

from tigarx.LocalSpline import LocallyConstructedSpline
from tigarx.common import mpirank
from tigarx.timeIntegration import GeneralizedAlphaIntegrator
from tigarx.RhinoTSplines import RhinoTSplineControlMesh

from tigarx.timing_util import perf_log
from tigarx.utils import interleave_and_expand


def dynamic_t_spline_local():
    FNAME = "sphere.iga"

    if not os.path.isfile(FNAME):
        if mpirank == 0:
            print(
                "ERROR: The required input file '"
                + FNAME
                + "' is not present in the working directory. "
                + "Please refer to the docstring at the top of this script."
            )
        exit()

    perf_log.start_timing("Dynamic t-spline local")
    perf_log.start_timing("Loading T-spline mesh")

    control_mesh = RhinoTSplineControlMesh(FNAME)

    perf_log.end_timing("Loading T-spline mesh")
    perf_log.start_timing("Generating spline space")

    # Define a function that, given a point, returns True if the point
    # is contained in the plate, but not the sphere.
    # (The plate is at $z=0$, and the sphere is in $z>0$.)
    def check_on_plate(x, on_boundary):
        return x[2] < np.finfo(default_real_type).eps

    spline = LocallyConstructedSpline.get_from_mesh_and_init(
        control_mesh, quad_degree=6, dofs_per_cp=3
    )
    spline.control_point_funcs[0].name = "FX"
    spline.control_point_funcs[1].name = "FY"
    spline.control_point_funcs[2].name = "FZ"
    spline.control_point_funcs[3].name = "FW"

    zero_dofs = control_mesh.getDofsByLocation(check_on_plate)
    zero_dofs = np.array(interleave_and_expand(zero_dofs, 3), dtype=np.int32)
    dofs_values = np.zeros(len(zero_dofs), dtype=np.float64)
    bcs = {"dirichlet": (zero_dofs, dofs_values)}

    perf_log.end_timing("Generating spline space")
    perf_log.start_timing("UFL problem setup")

    # The unknown midsurface displacement
    y_hom = dolfinx.fem.Function(spline.V)  # in homogeneous coordinates
    y_hom.name = "disp"

    # Quantities from the previous time step
    y_old_hom = dolfinx.fem.Function(spline.V)
    ydot_old_hom = dolfinx.fem.Function(spline.V)
    yddot_old_hom = dolfinx.fem.Function(spline.V)

    # Create a time integrator for the displacement.
    RHO_INF = 0.5
    DELTA_T = 0.001
    timeInt = GeneralizedAlphaIntegrator(
        RHO_INF, DELTA_T, y_hom, (y_old_hom, ydot_old_hom, yddot_old_hom)
    )

    # Get alpha-level quantities for use in the formulation.  (These are linear
    # combinations of old and new quantities.  The time integrator assumes that
    # they are in homogeneous representation.)
    y_alpha = spline.rationalize(timeInt.x_alpha())
    ydot_alpha = spline.rationalize(timeInt.xdot_alpha())
    yddot_alpha = spline.rationalize(timeInt.xddot_alpha())

    # The reference configuration is the mapping from parametric coordinates to
    # physical space.
    X = spline.F

    # The current configuration is defined at the alpha level in the formulation.
    x = X + y_alpha


    # Helper function to normalize a vector v.
    def unit(v):
        return v / ufl.sqrt(ufl.inner(v, v))


    # Helper function to compute geometric quantities for a midsurface
    # configuration x.
    def shellGeometry(x):

        # Covariant basis vectors:
        dxdxi = spline.parametricGrad(x)
        a0 = ufl.as_vector([dxdxi[0, 0], dxdxi[1, 0], dxdxi[2, 0]])
        a1 = ufl.as_vector([dxdxi[0, 1], dxdxi[1, 1], dxdxi[2, 1]])
        a2 = unit(ufl.cross(a0, a1))

        # Metric tensor:
        a = ufl.as_matrix(
            ((ufl.inner(a0, a0), ufl.inner(a0, a1)),
             (ufl.inner(a1, a0), ufl.inner(a1, a1)))
        )
        # Curvature:
        deriva2 = spline.parametricGrad(a2)
        b = -ufl.as_matrix(
            (
                (ufl.inner(a0, deriva2[:, 0]), ufl.inner(a0, deriva2[:, 1])),
                (ufl.inner(a1, deriva2[:, 0]), ufl.inner(a1, deriva2[:, 1])),
            )
        )

        return (a0, a1, a2, a, b)


    # Use the helper function to obtain shell geometry for the reference
    # and current configurations defined earlier.
    A0, A1, A2, A, B = shellGeometry(X)
    a0, a1, a2, a, b = shellGeometry(x)

    # Strain quantities.
    epsilon = 0.5 * (a - A)
    kappa = B - b


    # Helper function to convert a 2x2 tensor T to its local Cartesian
    # representation, in a shell configuration with metric a, and covariant
    # basis vectors a0 and a1.
    def cartesian(T, a, a0, a1):

        # Raise the indices on the curvilinear basis to obtain contravariant
        # basis vectors a0c and a1c.
        ac = ufl.inv(a)
        a0c = ac[0, 0] * a0 + ac[0, 1] * a1
        a1c = ac[1, 0] * a0 + ac[1, 1] * a1

        # Perform Gram--Schmidt orthonormalization to obtain the local Cartesian
        # basis vector e0 and e1.
        e0 = unit(a0)
        e1 = unit(a1 - e0 * ufl.inner(a1, e0))

        # Perform the change of basis on T and return the result.
        ea = ufl.as_matrix(
            (
                (ufl.inner(e0, a0c), ufl.inner(e0, a1c)),
                (ufl.inner(e1, a0c), ufl.inner(e1, a1c)),
            )
        )
        ae = ea.T
        return ea * T * ae


    # Use the helper function to compute the strain quantities in local
    # Cartesian coordinates.
    epsilonBar = cartesian(epsilon, A, A0, A1)
    kappaBar = cartesian(kappa, A, A0, A1)


    # Helper function to convert a 2x2 tensor to voigt notation, following the
    # convention for strains, where there is a factor of 2 applied to the last
    # component.
    def voigt(T):
        return ufl.as_vector([T[0, 0], T[1, 1], 2.0 * T[0, 1]])


    # The Young's modulus and Poisson ratio:
    E = 3e4
    nu = 0.3

    # The material matrix:
    D = (E / (1.0 - nu * nu)) * ufl.as_matrix(
        [[1.0, nu, 0.0], [nu, 1.0, 0.0], [0.0, 0.0, 0.5 * (1.0 - nu)]]
    )
    # The shell thickness:
    h_th = 0.03

    # Extension and bending resultants:
    nBar = h_th * D * voigt(epsilonBar)
    mBar = (h_th**3) * D * voigt(kappaBar) / 12.0

    # Compute the elastic potential energy density
    Wint = (
        0.5
        * (ufl.inner(voigt(epsilonBar), nBar) + ufl.inner(voigt(kappaBar), mBar))
        * spline.dx
    )

    # Take the Gateaux derivative of Wint(y_alpha) in the direction of the test
    # function z to obtain the internal virtual work.  Because y_alpha is not
    # a valid argument to derivative(), we take the derivative w.r.t. y_hom
    # instead, then scale by $1/\alpha_f$.
    z_hom = ufl.TestFunction(spline.V)
    z = spline.rationalize(z_hom)
    dWint = (1.0 / timeInt.ALPHA_F) * ufl.derivative(Wint, y_hom, z_hom)

    # Mass density:
    DENS = 10.0

    # Inertial contribution to the residual:
    dWmass = DENS * h_th * ufl.inner(yddot_alpha, z) * spline.dx

    # The penalty potential to model collision with the plate:
    PENALTY = 1e8
    gapFunction = ufl.conditional(ufl.lt(x[2], 0.0), -x[2], 0.0)
    contactForce = ufl.as_vector((0.0, 0.0, PENALTY * gapFunction))
    dWext = ufl.inner(-contactForce, z) * spline.dx

    # The full nonlinear residual for the shell problem:
    res = dWmass + dWint + dWext

    # Use derivative() to obtain the consistent tangent of the nonlinear residual,
    # considered as a function of displacement in homogeneous coordinates.
    dRes = ufl.derivative(res, y_hom)

    # Apply an initial condition to the sphere's velocity.
    # Note: Seems like the interpolation component by component
    # is the only option here.
    const_vec = [0.0, 0.0, -10.0]
    for field in range(len(const_vec)):
        sub_func = timeInt.xdot_old.sub(field)
        val = dolfinx.default_scalar_type(const_vec[field])
        sub_func.interpolate(lambda x: np.full((x.shape[1],), val))

    perf_log.end_timing("UFL problem setup")

    # Define files in which to accumulate time series for each component of the
    # displacement, and the geometry of the control mesh (which is needed for
    # visualization in ParaView).
    #
    # (Using letters x, y, and z instead of numbered components in the file names
    # makes loading time series in ParaView more straightforward.)

    # For x, y, and z components of displacement:
    vtx = dolfinx.io.VTXWriter(
        spline.mesh.comm,
        "results/disp.bp",
        [y_hom] + spline.control_point_funcs
    )

    for i in range(0, 50):

        if mpirank == 0:
            print(
                "------- Time step " + str(i + 1) +
                " , t = " + str(timeInt.t) + " -------"
            )
        perf_log.start_timing("Time for step " + str(i + 1))
        # Solve the nonlinear problem for this time step and put the solution
        # (in homogeneous coordinates) in y_hom.
        spline.solve_nonlinear_variational_problem(dRes, res, y_hom, bcs, rtol=1e-7)

        # Output fields needed for visualization.
        vtx.write(timeInt.t)

        # Advance to the next time step.
        timeInt.advance()

        perf_log.end_timing("Time for step " + str(i + 1))

    vtx.close()

    perf_log.end_timing("Dynamic t-spline local")

    ####### Postprocessing #######

    # Notes for plotting the results with ParaView:
    #
    # Load the time series from all seven files and combine them with the
    # Append Attributes filter.  Then use the Calculator filter to define the
    # vector field
    #
    # ((disp_X+FX)/FW-coordsX)*iHat+((disp_Y+FY)/FW-coordsY)*jHat+((disp_Z+FZ)/FW-coordsZ)*kHat
    #
    # which can then be used in the Warp by Vector filter.  Because the
    # parametric domain is artificially stretched out, the result of the Warp by
    # Vector filter will be much smaller, and the window will need to be re-sized
    # to fit the warped data.  The scale factor on the warp filter may need to
    # manually be set to 1.


if __name__ == "__main__":
    dynamic_t_spline_local()
