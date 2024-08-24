import numpy as np
import numba as nb

from tIGArx.BSplines import ExplicitBSplineControlMesh, uniform_knots
from tIGArx.utils import interleave_and_shift


def reference_dofs_1d(n_u, p, scalar_spline, bs=1):
    ref_dofs = np.zeros((n_u, (p + 1) * bs), dtype=np.int32)

    for i in range(0, n_u):
        xi = np.array([i / n_u + 1.0e-8])
        ref_dofs[i, :] = interleave_and_shift(
            scalar_spline.getNodes(xi), bs, scalar_spline.getNcp()
        )

    return ref_dofs


def reference_dofs_2d(n_u, n_v, p, q, scalar_spline, bs=1):
    ref_dofs = np.zeros((n_u * n_v, (p + 1) * (q + 1) * bs), dtype=np.int32)

    for j in range(0, n_v):
        for i in range(0, n_u):
            xi = np.array([i / n_u + 1.0e-8, j / n_v + 1.0e-8])
            ref_dofs[j * n_u + i, :] = interleave_and_shift(
                scalar_spline.getNodes(xi), bs, scalar_spline.getNcp()
            )

    return ref_dofs


def reference_dofs_3d(n_u, n_v, n_w, p, q, r, scalar_spline, bs=1):
    ref_dofs = np.zeros((n_u * n_v * n_w, (p + 1) * (q + 1) * (r + 1) * bs),
                        dtype=np.int32)

    for k in range(0, n_w):
        for j in range(0, n_v):
            for i in range(0, n_u):
                xi = np.array([i / n_u + 1.0e-8, j / n_v + 1.0e-8, k / n_w + 1.0e-8])
                ref_dofs[k * n_u * n_v + j * n_u + i, :] = interleave_and_shift(
                    scalar_spline.getNodes(xi), bs, scalar_spline.getNcp()
                )

    return ref_dofs


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
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


def get_extraction_ordering(cells, vertices, coords, spline, gdim=2):
    extraction_dofmap: np.ndarray

    # For 2D meshes the coordinate numbering does not match the
    # topology numbering. The correct ordering has to be obtained
    extraction_dofmap = np.zeros(len(cells), dtype=np.int32)
    for cell in cells:
        # Pick the center of interval/quad/hex for the evaluation point
        coord = coords[vertices[cell, :]].sum(axis=0) / 2 ** gdim
        extraction_dofmap[cell] = spline.getElement(coord)

    return extraction_dofmap


def test_bspline_dofs_1d():
    p = 3

    n_u = 9

    spline_mesh = ExplicitBSplineControlMesh(
        [p],
        [uniform_knots(p, 0.0, 1.0, n_u)],
    )
    scalar_spline = spline_mesh.getScalarSpline()
    cell_range = np.arange(n_u, dtype=np.int32)

    dofs = spline_mesh.getScalarSpline().getCpDofmap(cell_range)
    ref_dofs = reference_dofs_1d(n_u, p, scalar_spline)

    assert np.allclose(ref_dofs, dofs)

    dofs = spline_mesh.getScalarSpline().getCpDofmap(cell_range, 2)
    ref_dofs = reference_dofs_1d(n_u, p, scalar_spline, bs=2)

    assert np.allclose(ref_dofs, dofs)

    vect = [0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.4, 0.6, 0.6, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]
    spline_mesh = ExplicitBSplineControlMesh([3], [vect])
    scalar_spline = spline_mesh.getScalarSpline()

    ref_dofs = reference_dofs_1d(5, 3, scalar_spline)

    cell_range = np.arange(5, dtype=np.int32)
    dofs = scalar_spline.getCpDofmap(cell_range)

    assert np.allclose(ref_dofs, dofs)


def test_bspline_dofs_2d():
    p = 3
    q = 2

    n_u = 9
    n_v = 8

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v)
        ],
    )
    scalar_spline = spline_mesh.getScalarSpline()
    cell_range = np.arange(n_u * n_v, dtype=np.int32)

    dofs = scalar_spline.getCpDofmap(cell_range)
    ref_dofs = reference_dofs_2d(n_u, n_v, p, q, scalar_spline)

    assert np.allclose(ref_dofs, dofs)

    dofs = scalar_spline.getCpDofmap(cell_range, 2)
    ref_dofs = reference_dofs_2d(n_u, n_v, p, q, scalar_spline, bs=2)

    assert np.allclose(ref_dofs, dofs)


def test_bspline_dofs_3d():
    p = 3
    q = 2
    r = 4

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
    scalar_spline = spline_mesh.getScalarSpline()
    cell_range = np.arange(n_u * n_v * n_w, dtype=np.int32)

    dofs = scalar_spline.getCpDofmap(cell_range)
    ref_dofs = reference_dofs_3d(n_u, n_v, n_w, p, q, r, scalar_spline)

    assert np.allclose(ref_dofs, dofs)

    dofs = scalar_spline.getCpDofmap(cell_range, 3)
    ref_dofs = reference_dofs_3d(n_u, n_v, n_w, p, q, r, scalar_spline, bs=3)

    assert np.allclose(ref_dofs, dofs)


def test_bspline_csr_pre_alloc_1d():
    p = 3

    n_u = 9

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

    index_ptr, nnz_list = spline_mesh.getScalarSpline().getCSRPrealloc(2)

    bs = 2
    ref_dofmap = reference_dofs_1d(n_u, p, scalar_spline, bs=bs)
    rows = scalar_spline.getNcp() * bs
    max_dofs_per_row = (2 * p + 1) * bs

    index_ptr_ref, nnz_list_ref = get_csr_pre_allocation(
        np.arange(n_u, dtype=np.int32), ref_dofmap, rows, max_dofs_per_row
    )

    assert np.allclose(index_ptr, index_ptr_ref)
    assert np.allclose(nnz_list, nnz_list_ref)


def test_bspline_csr_pre_alloc_2d():
    p = 3
    q = 2

    n_u = 9
    n_v = 8

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

    index_ptr, nnz_list = spline_mesh.getScalarSpline().getCSRPrealloc(2)

    bs = 2
    ref_dofmap = reference_dofs_2d(n_u, n_v, p, q, scalar_spline, bs=bs)
    rows = scalar_spline.getNcp() * bs
    max_dofs_per_row = (2 * p + 1) * (2 * q + 1) * bs

    index_ptr_ref, nnz_list_ref = get_csr_pre_allocation(
        np.arange(n_u * n_v, dtype=np.int32), ref_dofmap, rows, max_dofs_per_row
    )

    assert np.allclose(index_ptr, index_ptr_ref)
    assert np.allclose(nnz_list, nnz_list_ref)


def test_bspline_csr_pre_alloc_3d():
    p = 3
    q = 2
    r = 4

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

    index_ptr, nnz_list = spline_mesh.getScalarSpline().getCSRPrealloc(3)

    bs = 3
    ref_dofmap = reference_dofs_3d(n_u, n_v, n_w, p, q, r, scalar_spline, bs=bs)
    rows = scalar_spline.getNcp() * bs
    max_dofs_per_row = (2 * p + 1) * (2 * q + 1) * (2 * r + 1) * bs

    index_ptr_ref, nnz_list_ref = get_csr_pre_allocation(
        np.arange(n_u * n_v * n_w, dtype=np.int32), ref_dofmap, rows, max_dofs_per_row
    )

    assert np.allclose(index_ptr, index_ptr_ref)
    assert np.allclose(nnz_list, nnz_list_ref)


def test_bspline_extraction_ordering():
    p = 3
    q = 2
    r = 4

    n_u = 6
    n_v = 5
    n_w = 4

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v)
        ],
    )
    scalar_spline = spline_mesh.getScalarSpline()

    mesh = scalar_spline.generateMesh()
    cells = np.arange(n_u * n_v, dtype=np.int32)
    vertices = mesh.geometry.dofmap.reshape(n_u * n_v, -1)
    coords = mesh.geometry.x

    ordering = scalar_spline.getExtractionOrdering(mesh)
    ref_ordering = get_extraction_ordering(cells, vertices, coords, scalar_spline, 2)

    assert np.allclose(ordering, ref_ordering)

    spline_mesh = ExplicitBSplineControlMesh(
        [p, q, r],
        [
            uniform_knots(p, 0.0, 1.0, n_u),
            uniform_knots(q, 0.0, 1.0, n_v),
            uniform_knots(r, 0.0, 1.0, n_w)
        ],
    )
    scalar_spline = spline_mesh.getScalarSpline()

    mesh = scalar_spline.generateMesh()
    cells = np.arange(n_u * n_v * n_w, dtype=np.int32)
    vertices = mesh.geometry.dofmap.reshape(n_u * n_v * n_w, -1)
    coords = mesh.geometry.x

    ordering = scalar_spline.getExtractionOrdering(mesh)
    ref_ordering = get_extraction_ordering(cells, vertices, coords, scalar_spline, 3)

    assert np.allclose(ordering, ref_ordering)
