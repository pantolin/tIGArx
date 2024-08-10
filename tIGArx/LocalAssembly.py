import ctypes

import numpy as np
import numba as nb

import dolfinx.fem.petsc

from cffi import FFI
from petsc4py import PETSc

from tIGArx.SplineInterface import AbstractScalarBasis
from tIGArx.utils import perf_log

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


def assemble_matrix(form: dolfinx.fem.Form, spline: AbstractScalarBasis):
    """
    Assemble matrix

    Args:
        form (dolfinx.fem.Form): form to assemble
        spline (AbstractScalarBasis): scalar basis

    Returns:
        A (dolfinx.cpp.la.PETScMatrix): assembled matrix
    """
    perf_log.start_timing("Assembling matrix", True)
    perf_log.start_timing("Getting form data")

    vertices, coords, gdim = get_vertices(form.mesh)

    cells = np.arange(form.mesh.topology.original_cell_index.size, dtype=np.int32)
    bs = form.function_spaces[0].dofmap.index_map_bs

    num_loc_points = 1
    for _ in range(gdim):
        num_loc_points *= (spline.getDegree() + 1)

    num_loc_dofs = num_loc_points * bs

    perf_log.end_timing("Getting form data")
    perf_log.start_timing("Creating dofmap")

    spline_dofmap = np.zeros((len(cells), num_loc_dofs), dtype=np.int32)
    extraction_dofmap = np.zeros(len(cells), dtype=np.int64)

    for cell in cells:
        # Pick the center of interval/quad/hex for the evaluation point
        coord = coords[vertices[cell, :]].sum(axis=0) / 2 ** gdim
        spline_dofmap[cell, :] = (spline.getNodes(coord))
        extraction_dofmap[cell] = spline.getElement(coord)

    perf_log.end_timing("Creating dofmap")
    perf_log.start_timing("Computing pre-allocation")

    max_per_row = (2 * spline.getDegree() + 1) ** gdim

    # nnz_per_row = get_nnz_pre_allocation(
    #     cells, spline_dofmap, spline.getNcp(), max_per_row
    # )

    ind_ptr, indices = get_csr_pre_allocation(
        cells, spline_dofmap, spline.getNcp(), max_per_row
    )

    perf_log.end_timing("Computing pre-allocation")
    perf_log.start_timing("Pre-allocating matrix")

    mat = PETSc.Mat(form.mesh.comm)
    # mat.createAIJ(spline.getNcp(), spline.getNcp(), nnz=nnz_per_row)

    mat.createAIJ(spline.getNcp(), spline.getNcp(), nnz=ind_ptr[-1])
    mat.setPreallocationCSR((ind_ptr, indices))

    perf_log.end_timing("Pre-allocating matrix")
    perf_log.start_timing("Packing constants")

    consts = dolfinx.cpp.fem.pack_constants(form._cpp_object)
    all_coeffs = dolfinx.cpp.fem.pack_coefficients(form._cpp_object)

    perf_log.end_timing("Packing constants")
    perf_log.start_timing("Computing extraction operators")

    extraction_operators = spline.get_lagrange_extraction_operators()

    perf_log.end_timing("Computing extraction operators")
    perf_log.start_timing("DOF permutation")

    permutation = get_lagrange_permutation(form, spline.getDegree(), gdim, bs)

    perf_log.end_timing("DOF permutation")
    perf_log.start_timing("Assembly step")

    for integral in form.integral_types:
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
                extraction_dofmap,
                permutation,
                bs,
                set_mat,
                PETSc.InsertMode.ADD_VALUES
            )

    mat.assemble()

    perf_log.end_timing("Assembly step")
    perf_log.end_timing("Assembling matrix")

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
    extraction_dofmap,
    permutation,
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

    for cell in cells:
        element = extraction_dofmap[cell]
        i = element // extraction_operators[0].shape[0]
        j = element % extraction_operators[0].shape[0]

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

        A_local = A_local[permutation, :][:, permutation]
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
    extraction_dofmap,
    permutation,
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

    for cell in cells:
        element = extraction_dofmap[cell]
        i = element // extraction_operators[0].shape[0]
        j = element % extraction_operators[0].shape[0]

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

        A_local = A_local[permutation, :][:, permutation]
        A_local = full_kron @ A_local @ full_kron.T

        set_vals(
            mat_handle,
            num_loc_dofs,
            pos.ctypes,
            num_loc_dofs,
            pos.ctypes,
            A_local.ctypes,
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
    perf_log.start_timing("Assembling vector", True)
    perf_log.start_timing("Getting form data")

    vertices, coords, gdim = get_vertices(form.mesh)

    cells = form.mesh.topology.original_cell_index
    bs = form.function_spaces[0].dofmap.index_map_bs
    num_loc_dofs = bs
    for _ in range(gdim):
        num_loc_dofs *= (spline.getDegree() + 1)

    perf_log.end_timing("Getting form data")
    perf_log.start_timing("Creating dofmap")

    spline_dofmap = np.zeros((len(cells), num_loc_dofs), dtype=np.int32)
    extraction_dofmap = np.zeros(len(cells), dtype=np.int64)

    for cell in cells:
        # Pick the center of interval/quad/hex for the evaluation point
        coord = coords[vertices[cell, :]].sum(axis=0) / 2 ** gdim
        spline_dofmap[cell, :] = (spline.getNodes(coord))
        extraction_dofmap[cell] = spline.getElement(coord)

    perf_log.end_timing("Creating dofmap")
    perf_log.start_timing("Allocating vector")

    vec = PETSc.Vec(form.mesh.comm)
    vec = vec.createWithArray(np.zeros(spline.getNcp()))

    perf_log.end_timing("Allocating vector")
    perf_log.start_timing("Packing constants")

    consts = dolfinx.cpp.fem.pack_constants(form._cpp_object)
    all_coeffs = dolfinx.cpp.fem.pack_coefficients(form._cpp_object)

    perf_log.end_timing("Packing constants")
    perf_log.start_timing("Computing extraction operators")

    extraction_operators = spline.get_lagrange_extraction_operators()

    perf_log.end_timing("Computing extraction operators")
    perf_log.start_timing("DOF permutation")

    permutation = get_lagrange_permutation(form, spline.getDegree(), gdim, bs)

    perf_log.end_timing("DOF permutation")
    perf_log.start_timing("Assembly step")

    for integral in form.integral_types:
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
                extraction_dofmap,
                permutation,
                bs,
                set_vec,
                PETSc.InsertMode.ADD_VALUES
            )

    vec.assemble()

    perf_log.end_timing("Assembly step")
    perf_log.end_timing("Assembling vector")

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
    extraction_dofmap,
    permutation,
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

    for cell in cells:
        element = extraction_dofmap[cell]
        i = element // extraction_operators[0].shape[0]
        j = element % extraction_operators[0].shape[0]

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

        vec_local = vec_local[permutation]
        vec_local = full_kron @ vec_local

        # vec.setValues(pos, vec_local, PETSc.InsertMode.ADD_VALUES)
        set_vals(
            vec,
            num_loc_dofs,
            pos.ctypes,
            vec_local.ctypes,
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


def get_lagrange_permutation(form: dolfinx.fem.Form, deg: int, gdim: int, bs: int = 1):
    """
    Get permutation for Lagrange basis

    Args:
        form (dolfinx.fem.Form): form object
        deg (int): degree of the basis
        gdim (int): mesh dimension`
        bs (int, optional): number of variables attached to each dof. Default is 1.
    Returns:
        permutation (np.array): permutation array
    """
    dof_coords = form.function_spaces[0].element.basix_element.points
    permutation = np.zeros(dof_coords.shape[0], dtype=np.uint32)
    
    if gdim == 1:
        for ind, coord in enumerate(dof_coords):
            index = int(coord[0] * deg)
            permutation[index] = ind

    elif gdim == 2:
        for ind, coord in enumerate(dof_coords):
            index_i = int(coord[0] * deg)
            index_j = int(coord[1] * deg)
            permutation[index_j * (deg + 1) + index_i] = ind
    
    elif gdim == 3:
        for ind, coord in enumerate(dof_coords):
            index_i = int(coord[0] * deg)
            index_j = int(coord[1] * deg)
            index_k = int(coord[2] * deg)
            permutation[index_k * (deg + 1) ** 2 + index_j * (deg + 1) + index_i] = ind
            
    else:
        raise ValueError("Invalid mesh dimension")
    
    return permutation.repeat(bs)


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
