import numpy as np

from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots
from tIGArx.LocalSpline import LocallyConstructedSpline
from tIGArx.MultiFieldSplines import EqualOrderSpline


def test_control_points_2d():
    p = 2
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

    local_spline = LocallyConstructedSpline.get_from_mesh_and_init(
        spline_mesh, quad_degree=2 * max(p, q), dofs_per_cp=1
    )

    ncp = spline_generator.getScalarSpline(-1).getNcp()
    ref_control_points = np.empty((ncp, 3), dtype=np.float64)
    for i in range(ncp):
        for d in range(3):
            ref_control_points[i, d] = spline_generator.getHomogeneousCoordinate(i, d)

    control_points = local_spline.control_points

    assert control_points.shape == ref_control_points.shape
    assert np.allclose(control_points, ref_control_points)


def test_control_points_3d():
    p = 2
    q = 3
    r = 4

    n_u = 6
    n_v = 5
    n_w = 4

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q, r],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v),
            uniform_knots(r, 0.0, 1.0, n_w)
        ],
    )

    spline_generator = EqualOrderSpline(1, spline_mesh)

    local_spline = LocallyConstructedSpline.get_from_mesh_and_init(
        spline_mesh, quad_degree=2 * max(p, q, r), dofs_per_cp=1
    )

    ncp = spline_generator.getScalarSpline(-1).getNcp()
    ref_control_points = np.empty((ncp, 4), dtype=np.float64)
    for i in range(ncp):
        for d in range(4):
            ref_control_points[i, d] = spline_generator.getHomogeneousCoordinate(i, d)

    control_points = local_spline.control_points

    assert control_points.shape == ref_control_points.shape
    assert np.allclose(control_points, ref_control_points)


def test_extracted_control_points_2d():
    p = 2
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

    local_spline = LocallyConstructedSpline.get_from_mesh_and_init(
        spline_mesh, quad_degree=2 * max(p, q), dofs_per_cp=1
    )

    ref_extracted_cps = spline_generator.cpFuncs
    extracted_cps = local_spline.control_point_funcs

    np.allclose(extracted_cps[0].x.array, ref_extracted_cps[0].x.array)
    np.allclose(extracted_cps[1].x.array, ref_extracted_cps[1].x.array)
    np.allclose(extracted_cps[2].x.array, ref_extracted_cps[2].x.array)


def test_extracted_control_points_3d():
    p = 2
    q = 3
    r = 4

    n_u = 6
    n_v = 5
    n_w = 4

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q, r],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v),
            uniform_knots(r, 0.0, 1.0, n_w)
        ],
    )

    spline_generator = EqualOrderSpline(1, spline_mesh)

    local_spline = LocallyConstructedSpline.get_from_mesh_and_init(
        spline_mesh, quad_degree=2 * max(p, q, r), dofs_per_cp=1
    )

    ref_extracted_cps = spline_generator.cpFuncs
    extracted_cps = local_spline.control_point_funcs

    np.allclose(extracted_cps[0].x.array, ref_extracted_cps[0].x.array)
    np.allclose(extracted_cps[1].x.array, ref_extracted_cps[1].x.array)
    np.allclose(extracted_cps[2].x.array, ref_extracted_cps[2].x.array)
    np.allclose(extracted_cps[3].x.array, ref_extracted_cps[3].x.array)
