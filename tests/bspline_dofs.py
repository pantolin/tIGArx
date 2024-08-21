import numpy as np

from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots


def reference_dofs_1d(n_u, p, scalar_spline):
    ref_dofs = np.zeros((n_u, p + 1), dtype=np.int32)

    for i in range(0, n_u):
        xi = np.array([i / n_u + 1.0e-8])
        ref_dofs[i, :] = scalar_spline.getNodes(xi)

    return ref_dofs


def reference_dofs_2d(n_u, n_v, p, q, scalar_spline):
    ref_dofs = np.zeros((n_u * n_v, (p + 1) * (q + 1)), dtype=np.int32)

    for j in range(0, n_v):
        for i in range(0, n_u):
            xi = np.array([i / n_u + 1.0e-8, j / n_v + 1.0e-8])
            ref_dofs[j * n_u + i, :] = scalar_spline.getNodes(xi)

    return ref_dofs


def reference_dofs_3d(n_u, n_v, n_w, p, q, r, scalar_spline):
    ref_dofs = np.zeros((n_u * n_v * n_w, (p + 1) * (q + 1) * (r + 1)),
                        dtype=np.int32)

    for k in range(0, n_w):
        for j in range(0, n_v):
            for i in range(0, n_u):
                xi = np.array([i / n_u + 1.0e-8, j / n_v + 1.0e-8, k / n_w + 1.0e-8])
                ref_dofs[k * n_u * n_v + j * n_u + i, :] = (
                    scalar_spline.getNodes(xi)
                )

    return ref_dofs


def get_csr_pre_allocation(cells, dofmap, rows, max_dofs_per_row):
    dofs_per_row = np.zeros((rows, max_dofs_per_row), dtype=np.int32)
    nnz_per_row = np.zeros(rows, dtype=np.int32)

    for cell in cells:
        for row_idx in dofmap[cell, :]:
            for dof in dofmap[cell, :]:
                found = False
                # Linear search is used here because maintaining a sorted
                # array is expected to be expensive
                for i in range(nnz_per_row[row_idx]):
                    if dofs_per_row[row_idx, i] == dof:
                        found = True
                        break
                if not found:
                    dofs_per_row[row_idx, nnz_per_row[row_idx]] = dof
                    nnz_per_row[row_idx] += 1

    index_ptr = np.zeros(rows + 1, dtype=np.int32)
    for i in range(rows):
        index_ptr[i + 1] = index_ptr[i] + nnz_per_row[i]

    indices = np.zeros(index_ptr[-1], dtype=np.int32)

    index = 0
    for row, row_dofs in enumerate(dofs_per_row):
        sorted_dofs = np.sort(row_dofs[:nnz_per_row[row]])

        for i in range(nnz_per_row[row]):
            indices[index] = sorted_dofs[i]
            index += 1

    return index_ptr, indices


def test_bspline_dofs():
    p = 3
    q = 2
    r = 4

    n_u = 8
    n_v = 8
    n_w = 8

    # First the uniform 1D case
    spline_mesh = ExplicitBSplineControlMesh(
        [p],
        [uniform_knots(p, 0.0, 1.0, n_u)],
    )
    scalar_spline = spline_mesh.getScalarSpline()

    ref_dofs = reference_dofs_1d(n_u, p, scalar_spline)

    dof_range = np.arange(n_u, dtype=np.int32)
    dofs = spline_mesh.getScalarSpline().getCpDofmap(dof_range)

    assert np.allclose(ref_dofs, dofs)

    vect = [0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.4, 0.6, 0.6, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
    spline_mesh = ExplicitBSplineControlMesh([3], [vect])
    scalar_spline = spline_mesh.getScalarSpline()

    ref_dofs = reference_dofs_1d(5, 3, scalar_spline)

    # for i in range(0, 5):
    #     xi = np.array([i / 5.0 + 1.0e-8])
    #     ref_dofs[i, :] = scalar_spline.getNodes(xi)

    dof_range = np.arange(5, dtype=np.int32)
    dofs = scalar_spline.getCpDofmap(dof_range)

    assert np.allclose(ref_dofs, dofs)

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v)
        ],
    )
    scalar_spline = spline_mesh.getScalarSpline()

    ref_dofs = reference_dofs_2d(n_u, n_v, p, q, scalar_spline)

    dof_range = np.arange(n_u * n_v, dtype=np.int32)
    dofs = scalar_spline.getCpDofmap(dof_range)

    assert np.allclose(ref_dofs, dofs)

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q, r],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v),
            uniform_knots(r, 0.0, 1.0, n_w)
        ],
    )
    scalar_spline = spline_mesh.getScalarSpline()

    ref_dofs = reference_dofs_3d(n_u, n_v, n_w, p, q, r, scalar_spline)

    dof_range = np.arange(n_u * n_v * n_w, dtype=np.int32)
    dofs = scalar_spline.getCpDofmap(dof_range)

    assert np.allclose(ref_dofs, dofs)


def test_bspline_csr_pre_alloc():

    p = 3
    q = 2
    r = 4

    n_u = 6
    n_v = 5
    n_w = 4

    # First the uniform 1D case
    spline_mesh = ExplicitBSplineControlMesh(
        [p],
        [uniform_knots(p, 0.0, 1.0, n_u)],
    )
    scalar_spline = spline_mesh.getScalarSpline()

    index_ptr, nnz_list = spline_mesh.getScalarSpline().getCSRPrealloc()

    ref_dofmap = reference_dofs_1d(n_u, p, scalar_spline)
    rows = scalar_spline.getNcp()
    max_dofs_per_row = 2 * p + 1

    index_ptr_ref, nnz_list_ref = get_csr_pre_allocation(
        np.arange(n_u, dtype=np.int32), ref_dofmap, rows, max_dofs_per_row
    )

    assert np.allclose(index_ptr, index_ptr_ref)
    assert np.allclose(nnz_list, nnz_list_ref)

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v)
        ],
    )
    scalar_spline = spline_mesh.getScalarSpline()

    index_ptr, nnz_list = spline_mesh.getScalarSpline().getCSRPrealloc()

    ref_dofmap = reference_dofs_2d(n_u, n_v, p, q, scalar_spline)
    rows = scalar_spline.getNcp()
    max_dofs_per_row = (2 * p + 1) * (2 * q + 1)

    index_ptr_ref, nnz_list_ref = get_csr_pre_allocation(
        np.arange(n_u * n_v, dtype=np.int32), ref_dofmap, rows, max_dofs_per_row
    )

    assert np.allclose(index_ptr, index_ptr_ref)
    assert np.allclose(nnz_list, nnz_list_ref)

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q, r],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v),
            uniform_knots(r, 0.0, 1.0, n_w)
        ],
    )
    scalar_spline = spline_mesh.getScalarSpline()

    index_ptr, nnz_list = spline_mesh.getScalarSpline().getCSRPrealloc()

    ref_dofmap = reference_dofs_3d(n_u, n_v, n_w, p, q, r, scalar_spline)
    rows = scalar_spline.getNcp()
    max_dofs_per_row = (2 * p + 1) * (2 * q + 1) * (2 * r + 1)

    index_ptr_ref, nnz_list_ref = get_csr_pre_allocation(
        np.arange(n_u * n_v * n_w, dtype=np.int32), ref_dofmap, rows, max_dofs_per_row
    )

    assert np.allclose(index_ptr, index_ptr_ref)
    assert np.allclose(nnz_list, nnz_list_ref)
