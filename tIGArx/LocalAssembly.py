import ctypes

import numpy as np
import numba as nb

import dolfinx
import dolfinx.fem.petsc
import ufl

from cffi import FFI
from petsc4py import PETSc

from tIGArx.SplineInterface import AbstractScalarBasis
from tIGArx.timing_util import perf_log

ffi = FFI()

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

options = {
    "cffi_extra_compile_args": [
        "-O3", "-march=native", "-mtune=native", "-ffast-math"
    ],
}


def solve_linear_variational_problem(
        lhs: ufl.form.Form,
        rhs: ufl.form.Form,
        spline: AbstractScalarBasis,
        bcs: dict[str, [np.ndarray, np.ndarray]],
        rtol=1e-12,
        profile=False,
) -> PETSc.Vec:
    """
    Solve the linear variational problem using the given forms and
    spline scalar basis. Returns the solution for control point
    coefficients in the form of a PETSc vector.

    Args:
        lhs (ufl.form.Form): left-hand side form
        rhs (ufl.form.Form): right-hand side form
        spline (AbstractScalarBasis): scalar basis
        bcs (dict[str, [np.ndarray, np.ndarray]]): boundary conditions
        profile (bool, optional): Flag to enable profiling information.
            Default is False.
        rtol (float, optional): relative tolerance for the solver.
            Default is 1e-12.

    Returns:
        PETSc.Vec: solution vector
    """
    if profile:
        perf_log.start_timing("Solving linear problem", True)
        perf_log.start_timing("Assembling problem", True)

    lhs_form = dolfinx.fem.form(lhs, jit_options=options)
    rhs_form = dolfinx.fem.form(rhs, jit_options=options)

    mat = assemble_matrix(lhs_form, spline, profile)
    vec = assemble_vector(rhs_form, spline, profile)

    # for i in range(mat.block_size):
    #     for j in range(mat.block_size):
    #         print(mat[i, j], end=" ")
    #     print()

    if profile:
        perf_log.end_timing("Assembling problem")
        perf_log.start_timing("Applying boundary conditions")

    # TODO - improve support for different types of boundary conditions
    for kind, bc in bcs.items():
        bc_pos, bc_vals = bc

        if kind == "dirichlet":
            mat.zeroRowsColumns(bc_pos, 1.0)
            vec.setValues(bc_pos, bc_vals, PETSc.InsertMode.INSERT_VALUES)
        else:
            raise ValueError("Unknown boundary condition type")

    if (mat.getDiagonal().array == 0).any():
        raise RuntimeError("Cannot solve a singular system")

    if profile:
        perf_log.end_timing("Applying boundary conditions")
        perf_log.start_timing("Solving problem")

    sol = ksp_solve_iteratively(mat, vec, rtol=rtol)

    if profile:
        perf_log.end_timing("Solving problem")
        perf_log.end_timing("Solving linear problem")

    return sol


def assemble_matrix(
        form: dolfinx.fem.Form, spline: AbstractScalarBasis, profile=False
) -> PETSc.Mat:
    """
    Assemble matrix

    Args:
        form (dolfinx.fem.Form): form to assemble
        spline (AbstractScalarBasis): scalar basis
        profile (bool, optional): Flag to enable profiling information.
            Default is False.

    Returns:
        A (PETSc.Mat): assembled matrix
    """
    return assembly_kernel(form, spline, profile)


def assemble_vector(
        form: dolfinx.fem.Form, spline: AbstractScalarBasis, profile=False
) -> PETSc.Vec:
    """
    Assemble matrix

    Args:
        form (dolfinx.fem.Form): form to assemble
        spline (AbstractScalarBasis): scalar basis

    Returns:
        A (PETSc.Vec): assembled vector
    """
    return assembly_kernel(form, spline, profile)


def assembly_kernel(
        form: dolfinx.fem.Form, spline: AbstractScalarBasis, profile=False
) -> PETSc.Mat | PETSc.Vec:
    """
    Assemble the matrix or vector using the given form and spline basis

    Args:
        form (dolfinx.fem.Form): form object
        spline (AbstractScalarBasis): scalar basis
        profile (bool, optional): Flag to enable profiling information. Default is False.

    Returns:
        PETSc.Mat | PETSc.Vec: The assembled matrix or vector
    """
    if profile:
        perf_log.start_timing(f"Assembling rank-{form.rank} form", True)
        perf_log.start_timing("Getting basic data")

    vertices, coords, gdim = get_vertices(form.mesh)

    cells = np.arange(form.mesh.topology.original_cell_index.size, dtype=np.int32)
    bs = form.function_spaces[0].dofmap.index_map_bs
    num_loc_dofs = bs
    for _ in range(gdim):
        num_loc_dofs *= (spline.getDegree() + 1)

    if profile:
        perf_log.end_timing("Getting basic data")
        perf_log.start_timing("Creating dofmap")

    spline_dofmap = np.zeros((len(cells), num_loc_dofs), dtype=np.int32)
    extraction_dofmap = np.zeros(len(cells), dtype=np.int64)

    for cell in cells:
        # Pick the center of interval/quad/hex for the evaluation point
        coord = coords[vertices[cell, :]].sum(axis=0) / 2 ** gdim

        spline_dofmap[cell, :] = interleave_and_shift(
            spline.getNodes(coord), bs, spline.getNcp()
        )
        extraction_dofmap[cell] = spline.getElement(coord)

    if profile:
        perf_log.end_timing("Creating dofmap")

    # The object that is passed to the assembly routines
    tensor: PETSc.Mat | PETSc.Vec

    dimension = spline.getNcp() * bs

    if form.rank == 2:
        if profile:
            perf_log.start_timing("Computing pre-allocation")

        max_per_row = bs * (2 * spline.getDegree() + 1) ** gdim

        ind_ptr, indices = get_csr_pre_allocation(
            cells, spline_dofmap, dimension, max_per_row
        )

        if profile:
            perf_log.end_timing("Computing pre-allocation")
            perf_log.start_timing("Allocating rank-2 tensor")

        tensor = PETSc.Mat(form.mesh.comm)
        tensor.createAIJ(dimension, dimension, nnz=ind_ptr[-1])
        tensor.setPreallocationCSR((ind_ptr, indices))

        if profile:
            perf_log.end_timing("Allocating rank-2 tensor")

    elif form.rank == 1:
        if profile:
            perf_log.start_timing("Allocating rank-1 tensor")

        tensor = PETSc.Vec(form.mesh.comm)
        tensor = tensor.createWithArray(np.zeros(dimension))

        if profile:
            perf_log.end_timing("Allocating rank-1 tensor")

    else:
        raise ValueError("Ranks other than 1 and 2 are not supported")

    if profile:
        perf_log.start_timing("Packing constants")

    consts = dolfinx.cpp.fem.pack_constants(form._cpp_object)
    all_coeffs = dolfinx.cpp.fem.pack_coefficients(form._cpp_object)

    if profile:
        perf_log.end_timing("Packing constants")
        perf_log.start_timing("Computing extraction operators")

    extraction_operators = spline.get_lagrange_extraction_operators()
    perm = get_lagrange_permutation(form, spline.getDegree(), gdim)
    permutation = interleave_and_expand(perm, bs)

    if profile:
        perf_log.end_timing("Computing extraction operators")
        perf_log.start_timing("Assembly step")

    for i, integral in enumerate(form.integral_types):
        if integral == dolfinx.fem.IntegralType.cell:
            if profile:
                perf_log.start_timing(f"Assembling integral {i}")

            kernel = getattr(
                form.ufcx_form.form_integrals[integral],
                "tabulate_tensor_float64",
            )
            coeffs = all_coeffs[(integral, -1)]

            if form.rank == 2:
                _assemble_matrix(
                    tensor.handle,
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
                    gdim,
                    set_mat,
                    PETSc.InsertMode.ADD_VALUES
                )

            elif form.rank == 1:
                _assemble_vector(
                    tensor.handle,
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
                    gdim,
                    set_vec,
                    PETSc.InsertMode.ADD_VALUES
                )

            if profile:
                perf_log.end_timing(f"Assembling integral {i}")

    if profile:
        perf_log.end_timing("Assembly step")
        perf_log.end_timing(f"Assembling rank-{form.rank} form")

    if form.rank == 2 or form.rank == 1:
        tensor.assemble()

    return tensor


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _assemble_matrix(
        mat_handle,
        kernel,
        vertices,
        coords,
        dofmap,
        num_loc_dofs,
        coeffs,
        consts,
        cells,
        operators,
        extraction_dofmap,
        permutation,
        bs,
        gdim,
        set_vals,
        mode
):
    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    # Allocating memory for the local matrix and the full Kronecker product
    mat_local = np.zeros((num_loc_dofs, num_loc_dofs), dtype=PETSc.ScalarType)
    full_kron = np.zeros((num_loc_dofs, num_loc_dofs), dtype=PETSc.ScalarType)

    # Matrix for the number of variables attached to each dof
    bs_mat = np.eye(bs, dtype=PETSc.ScalarType)

    for cell in cells:
        element = extraction_dofmap[cell]

        if gdim == 1 or len(operators) == 1:
            full_kron[:, :] = np.kron(operators[0][element], bs_mat)

        elif gdim == 2:
            j = element // operators[0].shape[0]
            i = element % operators[0].shape[0]

            full_kron[:, :] = np.kron(
                np.kron(
                    operators[1][j],
                    operators[0][i]
                ),
                bs_mat
            )
        else:
            k = element // (operators[0].shape[0] * operators[1].shape[0])
            ij = element % (operators[0].shape[0] * operators[1].shape[0])
            j = ij // operators[0].shape[0]
            i = ij % operators[0].shape[0]

            full_kron[:, :] = np.kron(
                np.kron(
                    np.kron(
                        operators[2][k],
                        operators[1][j]
                    ),
                    operators[0][i]
                ),
                bs_mat,
            )

        pos = dofmap[cell, :]
        cell_coords[:, :] = coords[vertices[cell, :]]
        mat_local.fill(0.0)

        kernel(
            ffi.from_buffer(mat_local),
            ffi.from_buffer(coeffs[cell]),
            ffi.from_buffer(consts),
            ffi.from_buffer(cell_coords),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
        )

        # Permute the rows and columns so that they match the
        # indexing of the spline basis
        mat_local[:, :] = mat_local[permutation, :][:, permutation]
        mat_local[:, :] = full_kron @ mat_local @ full_kron.T

        # mat_handle.setValues(pos, pos, mat_local, PETSc.InsertMode.ADD_VALUES)
        set_vals(
            mat_handle,
            num_loc_dofs,
            pos.ctypes,
            num_loc_dofs,
            pos.ctypes,
            mat_local.ctypes,
            mode
        )


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
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
        operators,
        extraction_dofmap,
        permutation,
        bs,
        gdim,
        set_vals,
        mode
):
    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    # Allocating memory for the local vector and the full Kronecker product
    vec_local = np.zeros(num_loc_dofs, dtype=PETSc.ScalarType)
    full_kron = np.zeros((num_loc_dofs, num_loc_dofs), dtype=PETSc.ScalarType)

    # Matrix for the number of variables attached to each dof
    bs_mat = np.eye(bs, dtype=PETSc.ScalarType)

    for cell in cells:
        element = extraction_dofmap[cell]

        if gdim == 1 or len(operators) == 1:
            full_kron[:, :] = np.kron(operators[0][element], bs_mat)

        elif gdim == 2:
            j = element // operators[0].shape[0]
            i = element % operators[0].shape[0]

            full_kron[:, :] = np.kron(
                np.kron(
                    operators[1][j],
                    operators[0][i]
                ),
                bs_mat
            )
        else:
            k = element // (operators[0].shape[0] * operators[1].shape[0])
            ij = element % (operators[0].shape[0] * operators[1].shape[0])
            j = ij // operators[0].shape[0]
            i = ij % operators[0].shape[0]

            full_kron[:, :] = np.kron(
                np.kron(
                    np.kron(
                        operators[2][k],
                        operators[1][j]
                    ),
                    operators[0][i]
                ),
                bs_mat,
            )

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

        vec_local[:] = vec_local[permutation]
        vec_local[:] = full_kron @ vec_local

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


def get_lagrange_permutation(form: dolfinx.fem.Form, deg: int, gdim: int):
    """
    Get permutation for Lagrange basis

    Args:
        form (dolfinx.fem.Form): form object
        deg (int): degree of the basis
        gdim (int): mesh dimension`
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
            # TODO - investigate why j goes before i here
            permutation[index_i * (deg + 1) + index_j] = ind

    elif gdim == 3:
        for ind, coord in enumerate(dof_coords):
            index_i = int(coord[0] * deg)
            index_j = int(coord[1] * deg)
            index_k = int(coord[2] * deg)
            permutation[index_k * (deg + 1) ** 2 + index_j * (deg + 1) + index_i] = ind

    else:
        raise ValueError("Invalid mesh dimension")

    return permutation


def stack_and_shift(arr: np.ndarray, repeats: int, shift: int) -> np.ndarray:
    """
    Make ``repeats`` copies of the array ``arr`` and stack them
    after adding a shift to each successive copy. Used to expand
    the number of degrees of freedom attached to each variable in
    a blocked manner, by appending shifted copies of the array.

    Args:
        arr (np.ndarray): array to repeat
        repeats (int): number of repeats
        shift (int): shift value

    Returns:
        np.ndarray: stacked and shifted array
    """
    shifts = np.arange(repeats) * shift
    shifts = np.repeat(shifts, len(arr))

    repeated_arr = np.tile(arr, repeats)

    return repeated_arr + shifts


def interleave_and_shift(arr: np.ndarray, n: int, shift: int) -> np.ndarray:
    """
    Repeat each element of ``arr`` ``n`` times and each
    successive element is shifted by ``shift``. Used to expand
    the number of degrees of freedom attached to each variable
    in a blocked manner while keeping the their order.

    Args:
        arr (np.ndarray): array to repeat
        n (int): number of repeats
        shift (int): shift value

    Returns:
        np.ndarray: interleaved and shifted array
    """
    repeated_values = np.repeat(arr, n)
    increments = np.tile(np.arange(n) * shift, len(arr))

    return repeated_values + increments


def interleave_and_expand(arr: np.ndarray, n: int) -> np.ndarray:
    """
    Repeat each element of ``arr`` ``n`` times and multiply
    all of them by ``n``. Successive elements are then
    incremented. Used to expand the number of degrees of
    freedom attached to each variable in a contiguous manner.

    Args:
        arr (np.ndarray): array to repeat
        n (int): number of repeats

    Returns:
        np.ndarray: interleaved and incremented array
    """
    repeated_values = np.repeat(arr, n)
    increments = np.tile(np.arange(n), len(arr))

    return repeated_values * n + increments


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
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


def ksp_solve_iteratively(A: PETSc.Mat, b: PETSc.Vec, debug=False, rtol=1e-12):
    """
    Solve the linear system Ax = b using Conjugate Gradient
    and block JACOBI preconditioning.

    Args:
        A (PETSc.Mat): The system matrix.
        b (PETSc.Vec): The right-hand side vector.
        rtol (float, optional): The relative tolerance for the solver.
            Default is 1e-12.
        profile (bool, optional): Flag to enable profiling information.
            Default is False.
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
    if debug:
        print("-" * 60)
        print("Using CG solver with BJACOBI preconditioning")
        print(f"Matrix size:            {A.getSize()[0]}")
        info = A.getInfo()
        print(f"No. of non-zeros:       {info['nz_used']}")
        timer = dolfinx.common.Timer()
        timer.start()

    ksp.solve(b, vec)

    if debug:
        print(f"Solve took:             {timer.stop()}")
        print("-" * 60)

    vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD,
    )

    return vec


def dolfinx_assemble_linear_variational_problem(
        lhs: ufl.form.Form,
        rhs: ufl.form.Form,
        profile=False,
) -> PETSc.Vec:
    """
    Test of reference dolfinx tensor assembly time
    """
    if profile:
        perf_log.start_timing("Dolfinx assembly", True)

    lhs_form = dolfinx.fem.form(lhs, jit_options=options)
    rhs_form = dolfinx.fem.form(rhs, jit_options=options)

    mat = dolfinx.fem.assemble_matrix(lhs_form)
    vec = dolfinx.fem.assemble_vector(rhs_form)

    if profile:
        perf_log.end_timing("Dolfinx assembly")

    return mat, vec
