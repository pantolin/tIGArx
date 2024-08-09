import ctypes

import numpy as np
import numba as nb
import petsc4py.PETSc

import dolfinx.fem.petsc

from cffi import FFI
from petsc4py import PETSc

from tIGArx.SplineInterface import AbstractScalarBasis

ffi = FFI()

import dolfinx

petsc_lib = dolfinx.fem.petsc.load_petsc_lib(ctypes.cdll.LoadLibrary)

set_mat = getattr(petsc_lib, "MatSetValuesLocal")
set_vec = getattr(petsc_lib, "VecSetValuesLocal")

set_mat.restype = None
set_mat.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]

set_vec.restype = None
set_vec.argtypes = [
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]


def assemble_matrix(form, spline: AbstractScalarBasis):
    """
    Assemble matrix

    Args:
        form (dolfinx.fem.Form): form to assemble
        spline (AbstractScalarBasis): scalar basis

    Returns:
        A (dolfinx.cpp.la.PETScMatrix): assembled matrix
    """

    vertices, coords, gdim = get_vertices(form.mesh)

    cells = form.mesh.topology.original_cell_index
    bs = form.function_spaces[0].dofmap.index_map_bs
    num_loc_dofs = bs
    for _ in range(gdim):
        num_loc_dofs *= (spline.getDegree() + 1)

    spline_dofmap = np.zeros((len(cells), num_loc_dofs), dtype=np.int32)

    for cell in cells:
        # Pick the center of interval/quad/hex for the evaluation point
        coord = coords[vertices[cell, :]].sum(axis=0) / 2 ** gdim
        spline_dofmap[cell, :] = (spline.getNodes(coord))

    max_per_row = (2 * spline.getDegree() + 1) ** gdim

    # nnz_per_row = get_nnz_pre_allocation(
    #     cells, spline_dofmap, spline.getNcp(), max_per_row
    # )

    ind_ptr, indices = get_csr_pre_allocation(
        cells, spline_dofmap, spline.getNcp(), max_per_row
    )

    mat = PETSc.Mat(form.mesh.comm)
    mat.createAIJ(spline.getNcp(), spline.getNcp(), nnz=ind_ptr[-1])
    mat.setPreallocationCSR((ind_ptr, indices))

    consts = dolfinx.cpp.fem.pack_constants(form._cpp_object)
    all_coeffs = dolfinx.cpp.fem.pack_coefficients(form._cpp_object)

    integrals = form.integral_types

    extraction_operators = spline.get_lagrange_extraction_operators()

    for integral in integrals:
        if integral == dolfinx.fem.IntegralType.cell:

            kernel = getattr(
                form.ufcx_form.form_integrals[integral],
                "tabulate_tensor_float64",
            )
            coeffs = all_coeffs[(integral, -1)]

            assemble_cells(
                mat.handle,
                kernel,
                vertices,
                coords,
                spline_dofmap,
                num_loc_dofs,
                coeffs,
                consts,
                cells,
                extraction_operators,
                bs,
                set_mat,
                PETSc.InsertMode.ADD_VALUES
            )

    mat.assemble()

    return mat


def _assemble_cells(
    mat,
    kernel,
    vertices,
    coords,
    dofmap,
    num_loc_dofs,
    coeffs,
    consts,
    cells,
    extraction_operators,
    bs,
):
    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    A_local = np.zeros((num_loc_dofs, num_loc_dofs), dtype=PETSc.ScalarType)
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    bs_mat = np.ones(bs, dtype=np.float64)

    for k, cell in enumerate(cells):
        j = cell // extraction_operators[0].shape[0]
        i = cell % extraction_operators[0].shape[0]

        extraction_i = np.ascontiguousarray(extraction_operators[0][i, :, :])
        extraction_j = np.ascontiguousarray(extraction_operators[1][j, :, :])

        extraction_kron = np.kron(extraction_i, extraction_j)
        full_kron = np.kron(extraction_kron, bs_mat)

        pos = dofmap[cell, :]
        cell_coords[:, :] = coords[vertices[cell, :]]
        A_local.fill(0.0)

        kernel(
            ffi.from_buffer(A_local),
            ffi.from_buffer(coeffs[cell]),
            ffi.from_buffer(consts),
            ffi.from_buffer(cell_coords),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
        )

        A_local = full_kron @ A_local @ full_kron.T

        mat.setValues(pos, pos, A_local, PETSc.InsertMode.ADD_VALUES)


@nb.njit
def assemble_cells(
    mat_handle,
    kernel,
    vertices,
    coords,
    dofmap,
    num_loc_dofs,
    coeffs,
    consts,
    cells,
    extraction_operators,
    bs,
    set_vals,
    mode
):
    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    A_local = np.zeros((num_loc_dofs, num_loc_dofs), dtype=PETSc.ScalarType)
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    bs_mat = np.ones(bs, dtype=np.float64)

    for k, cell in enumerate(cells):
        j = cell // extraction_operators[0].shape[0]
        i = cell % extraction_operators[0].shape[0]

        extraction_i = np.ascontiguousarray(extraction_operators[0][i, :, :])
        extraction_j = np.ascontiguousarray(extraction_operators[1][j, :, :])

        extraction_kron = np.kron(extraction_i, extraction_j)
        full_kron = np.kron(extraction_kron, bs_mat)

        pos = dofmap[cell, :]
        cell_coords[:, :] = coords[vertices[cell, :]]
        A_local.fill(0.0)

        kernel(
            ffi.from_buffer(A_local),
            ffi.from_buffer(coeffs[cell]),
            ffi.from_buffer(consts),
            ffi.from_buffer(cell_coords),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
        )

        A_local = full_kron @ A_local @ full_kron.T

        # Using ffi here as a "hack" to avoid ctypes python function
        set_vals(
            mat_handle,
            num_loc_dofs,
            ffi.from_buffer(pos),
            num_loc_dofs,
            ffi.from_buffer(pos),
            ffi.from_buffer(A_local),
            mode
        )


def assemble_vector(form, spline: AbstractScalarBasis):
    """
    Assemble matrix

    Args:
        form (dolfinx.fem.Form): form to assemble
        spline (AbstractScalarBasis): scalar basis

    Returns:
        A (dolfinx.cpp.la.PETScMatrix): assembled matrix
    """

    vertices, coords, gdim = get_vertices(form.mesh)

    cells = form.mesh.topology.original_cell_index
    bs = form.function_spaces[0].dofmap.index_map_bs
    num_loc_dofs = bs
    for _ in range(gdim):
        num_loc_dofs *= (spline.getDegree() + 1)

    spline_dofmap = np.zeros((len(cells), num_loc_dofs), dtype=np.int32)

    for cell in cells:
        # Pick the center of interval/quad/hex for the evaluation point
        coord = coords[vertices[cell, :]].sum(axis=0) / 2 ** gdim
        spline_dofmap[cell, :] = (spline.getNodes(coord))

    vec = PETSc.Vec(form.mesh.comm)
    vec = vec.createWithArray(np.zeros(spline.getNcp()))

    consts = dolfinx.cpp.fem.pack_constants(form._cpp_object)
    all_coeffs = dolfinx.cpp.fem.pack_coefficients(form._cpp_object)

    integrals = form.integral_types

    extraction_operators = spline.get_lagrange_extraction_operators()

    for integral in integrals:
        if integral == dolfinx.fem.IntegralType.cell:

            kernel = getattr(
                form.ufcx_form.form_integrals[integral],
                "tabulate_tensor_float64",
            )
            coeffs = all_coeffs[(integral, -1)]

            _assemble_vector(
                vec.handle,
                kernel,
                vertices,
                coords,
                spline_dofmap,
                num_loc_dofs,
                coeffs,
                consts,
                cells,
                extraction_operators,
                bs,
                set_vec,
                PETSc.InsertMode.ADD_VALUES
            )

    vec.assemble()

    return vec


@nb.njit
def _assemble_vector(
    vec,
    kernel,
    vertices,
    coords,
    dofmap,
    num_loc_dofs,
    coeffs,
    consts,
    cells,
    extraction_operators,
    bs,
    set_vals,
    mode
):
    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    vec_local = np.zeros(num_loc_dofs, dtype=PETSc.ScalarType)
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)
    # Matrix for the number of variables attached to each dof
    bs_mat = np.ones(bs, dtype=np.float64)

    for k, cell in enumerate(cells):
        j = cell // extraction_operators[0].shape[0]
        i = cell % extraction_operators[0].shape[0]

        extraction_i = np.ascontiguousarray(extraction_operators[0][i, :, :])
        extraction_j = np.ascontiguousarray(extraction_operators[1][j, :, :])

        extraction_kron = np.kron(extraction_i, extraction_j)
        full_kron = np.kron(extraction_kron, bs_mat)

        pos = dofmap[cell, :]
        cell_coords[:, :] = coords[vertices[cell, :]]
        vec_local.fill(0.0)

        kernel(
            ffi.from_buffer(vec_local),
            ffi.from_buffer(coeffs[cell]),
            ffi.from_buffer(consts),
            ffi.from_buffer(cell_coords),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
        )

        vec_local = full_kron @ vec_local

        # vec.setValues(pos, vec_local, PETSc.InsertMode.ADD_VALUES)
        set_vals(
            vec,
            num_loc_dofs,
            ffi.from_buffer(pos),
            ffi.from_buffer(vec_local),
            mode
        )


def get_vertices(mesh: dolfinx.mesh.Mesh):
    """
    Get mesh vertices

    Args:
        mesh (dolfinx.mesh.Mesh): mesh object
    Returns:
        vertices (np.array(np.int32)): vertices associated to each cell
        coords (np.array): mesh coordinates
        gdim (np.int32): mesh dimension
    """
    coords = mesh.geometry.x
    gdim = mesh.geometry.dim
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    vertices = mesh.geometry.dofmap.reshape(num_cells, -1)

    return vertices, coords, gdim


def lock_inactive_dofs(inactive_dofs: list[np.int32], A: PETSc.Mat, alpha: float):
    """
    Lock inactive degrees of freedom and checks that all diagonal entries are > 0

    Args:
        inactive_dofs (list[np.int32]): list of inactive degrees of freedom
        A (PETSc.Mat): matrix on which to apply the locking
        alpha (float): value for the diagonal entries

    Returns:
        A (PETSc.Mat): matrix on which the locking has been applied
    """

    A.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)

    # print(f"Zeroing {len(inactive_dofs)} rows and columns")
    A.zeroRowsColumns(inactive_dofs, diag=alpha)

    A.assemble()

    # check diagonal
    ad = A.getDiagonal()
    if (ad.array == 0).any():
        raise RuntimeError("Zeros on the diagonal should not happen")

    return A


@nb.njit
def get_nnz_pre_allocation_small(cells, dofmap, rows):
    # Use a boolean matrix to track nonzero entries
    nnz_bool = np.zeros((rows, rows), dtype=np.uint8)

    # Iterate over cells and dofmap to populate the boolean matrix
    for cell in cells:
        for row_idx in dofmap[cell, :]:
            for dof in dofmap[cell, :]:
                nnz_bool[row_idx, dof] = 1

    # Sum across rows to get nnz per row
    nnz_per_row = nnz_bool.sum(axis=1)

    return nnz_per_row.astype(np.int32)


@nb.njit
def get_nnz_pre_allocation(cells, dofmap, rows, max_dofs_per_row):
    temp_dofs = np.zeros((rows, max_dofs_per_row), dtype=np.int32)
    nnz_per_row = np.zeros(rows, dtype=np.int32)

    for cell in cells:
        for row_idx in dofmap[cell, :]:
            for dof in dofmap[cell, :]:
                found = False
                # Linear search is used here because maintaining a sorted
                # array is expected to be expensive
                for i in range(nnz_per_row[row_idx]):
                    if temp_dofs[row_idx, i] == dof:
                        found = True
                        break
                if not found:
                    temp_dofs[row_idx, nnz_per_row[row_idx]] = dof
                    nnz_per_row[row_idx] += 1

    return nnz_per_row


@nb.njit
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


def ksp_solve_direct(A: PETSc.Mat, b: PETSc.Vec, profile=False):
    """
    Solve a linear system

    Args:
        A (PETSc.Mat): matrix
        b (PETSc.Vec): right hand side

    Returns:
        vec (PETSc.Vec): A @ vec = b
    """

    # Direct solver using mumps
    ksp = PETSc.KSP().create(A.comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    ksp.getPC().setType("lu")
    ksp.getPC().setFactorSolverType("mumps")
    vec = b.copy()

    if profile:
        print("-" * 60)
        print("Using direct solver (MUMPS)")
        print(f"Matrix size:            {A.size[0]}")
        info = A.getInfo()
        print("No. of non-zeros:      ", info["nz_used"])
        timer = dolfinx.common.Timer()

    ksp.solve(b, vec)

    if profile:
        print(f"Solve took:             {timer.elapsed()[0]}")
        print("-" * 60)

    vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD,
    )
    return vec


def ksp_solve_iteratively(A: PETSc.Mat, b: PETSc.Vec, profile=False, rtol=1e-12):
    """
    Solve the linear system Ax = b using Conjugate Gradient and block JACOBI preconditioning.

    Args:
        A (PETSc.Mat): The system matrix.
        b (PETSc.Vec): The right-hand side vector.
        rtol (float, optional): The relative tolerance for the solver. Default is 1e-12.
        profile (bool, optional): Flag to enable profiling information. Default is False.
    Returns:
        PETSc.Vec: The solution vector.
    """
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.BJACOBI)

    ksp.setTolerances(rtol=rtol)

    vec = b.copy()
    if profile:
        print("-" * 60)
        print("Using CG solver with BJACOBI preconditioning")
        print(f"Matrix size:            {A.getSize()[0]}")
        info = A.getInfo()
        print(f"No. of non-zeros:       {info['nz_used']}")
        timer = dolfinx.common.Timer()
        timer.start()

    ksp.solve(b, vec)

    if profile:
        print(f"Solve took:             {timer.stop()}")
        print("-" * 60)

    vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD,
    )

    return vec