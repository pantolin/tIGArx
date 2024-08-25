import numpy as np

from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots
from tIGArx.ExtractedSpline import ExtractedSpline
from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.MultiFieldSplines import EqualOrderSpline


def test_control_points_2d():
    p = 3
    q = 3

    n_u = 5
    n_v = 5

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v)
        ],
    )

    spline_generator = EqualOrderSpline(1, spline_mesh)

    local_spline = LocallyConstructedSpline(
        spline_mesh, quad_degree=2 * p, dofs_per_cp=1
    )
    local_spline.init_extracted_control_points()

    ncp = spline_generator.getScalarSpline(-1).getNcp()
    ref_control_points = np.empty((ncp, 3), dtype=np.float64)
    for i in range(ncp):
        for d in range(3):
            ref_control_points[i, d] = spline_generator.getHomogeneousCoordinate(i, d)

    control_points = local_spline.control_points

    assert control_points.shape == ref_control_points.shape
    assert np.allclose(control_points, ref_control_points)


def test_control_points_3d():
    p = 3
    q = 3
    r = 3

    n_u = 9
    n_v = 8
    n_w = 7

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q, r],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v),
            uniform_knots(r, 0.0, 1.0, n_w)
        ],
    )

    spline_generator = EqualOrderSpline(1, spline_mesh)

    local_spline = LocallyConstructedSpline(
        spline_mesh, quad_degree=3 * p, dofs_per_cp=1
    )
    local_spline.init_extracted_control_points()

    ncp = spline_generator.getScalarSpline(-1).getNcp()
    ref_control_points = np.empty((ncp, 4), dtype=np.float64)
    for i in range(ncp):
        for d in range(4):
            ref_control_points[i, d] = spline_generator.getHomogeneousCoordinate(i, d)

    control_points = local_spline.control_points

    assert control_points.shape == ref_control_points.shape
    assert np.allclose(control_points, ref_control_points)


def test_cp_func_vectors():
    p = 3
    q = 3

    n_u = 9
    n_v = 8

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v)
        ],
    )

    spline_generator = EqualOrderSpline(1, spline_mesh)

    local_spline = LocallyConstructedSpline(
        spline_mesh, quad_degree=2 * p, dofs_per_cp=1
    )

    ref_extracted_cps = spline_generator.cpFuncs

    local_spline.init_extracted_control_points()

    extracted_cps = local_spline.control_point_funcs

    print(extracted_cps)
