import ctypes

import numpy as np
import numba as nb

import dolfinx
import dolfinx.fem.petsc

from cffi import FFI
from petsc4py import PETSc

from tIGArx.SplineInterface import AbstractScalarBasis
from tIGArx.timing_util import perf_log
from tIGArx.utils import interleave_and_expand, get_lagrange_permutation

# Initialization of the FFI interface required to call PETSc functions
# from inside numba JIT-compiled functions
ffi = FFI()

petsc_lib = dolfinx.fem.petsc.load_petsc_lib(ctypes.cdll.LoadLibrary)

# The functions exist in the lib but their signatures are not exposed
# so we have to define them manually
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


def assemble_matrix(
        form: dolfinx.fem.Form,
        spline: AbstractScalarBasis,
        mat: PETSc.Mat | None = None,
        profile=False
) -> PETSc.Mat:
    """
    Assemble the matrix from a given form. Allocates the matrix if it is not
    provided. Returns the assembled matrix.

    Args:
        form (dolfinx.fem.Form): form to assemble
        spline (AbstractScalarBasis): scalar basis
        mat (PETSc.Mat, optional): matrix to assemble into. Default is None.
        profile (bool, optional): Flag to enable profiling information.
            Default is False.

    Returns:
        A (PETSc.Mat): assembled matrix
    """
    if mat is None:
        if profile:
            perf_log.start_timing("Computing pre-allocation")
        bs = form.function_spaces[0].dofmap.index_map_bs
        csr_allocation = spline.getCSRPrealloc(block_size=bs)

        if profile:
            perf_log.end_timing("Computing pre-allocation")
            perf_log.start_timing("Allocating matrix")

        mat = PETSc.Mat(form.mesh.comm)
        mat.createAIJ(
            spline.getNcp() * bs,
            spline.getNcp() * bs,
            csr=csr_allocation
        )

        if profile:
            perf_log.end_timing("Allocating matrix")
    else:
        mat.zeroEntries()


    assembly_kernel(form, spline, mat, profile)
    mat.assemble()

    return mat


def assemble_vector(
        form: dolfinx.fem.Form,
        spline: AbstractScalarBasis,
        vec: PETSc.Vec | None = None,
        profile=False
) -> PETSc.Vec:
    """
    Assemble the vector from a given form. Allocates the vector if it is not
    provided. Returns the assembled vector.

    Args:
        form (dolfinx.fem.Form): form to assemble
        spline (AbstractScalarBasis): scalar basis
        vec (PETSc.Vec, optional): vector to assemble into. Default is None.
        profile (bool, optional): Flag to enable profiling information.
            Default is False.

    Returns:
        A (PETSc.Vec): assembled vector
    """
    if vec is None:
        if profile:
            perf_log.start_timing("Allocating vector")

        vec = PETSc.Vec(form.mesh.comm)
        vec.createWithArray(
            np.zeros(spline.getNcp() * form.function_spaces[0].dofmap.index_map_bs)
        )

        if profile:
            perf_log.end_timing("Allocating vector")
    else:
        vec.zeroEntries()

    assembly_kernel(form, spline, vec, profile)
    vec.assemble()

    return vec


def assembly_kernel(
        form: dolfinx.fem.Form,
        spline: AbstractScalarBasis,
        tensor: PETSc.Mat | PETSc.Vec,
        profile=False
):
    """
    Assemble the matrix or vector using the given form and spline basis

    Args:
        form (dolfinx.fem.Form): form object
        spline (AbstractScalarBasis): scalar basis
        tensor (PETSc.Mat | PETSc.Vec): matrix or vector to assemble into
        profile (bool, optional): Flag to enable profiling information.
            Default is False.
    """
    if profile:
        perf_log.start_timing(f"Assembling rank-{form.rank} form", True)
        perf_log.start_timing("Getting basic data")

    coords = form.mesh.geometry.x
    gdim = form.mesh.geometry.dim

    num_cells = form.mesh.topology.index_map(form.mesh.topology.dim).size_local
    vertices = form.mesh.geometry.dofmap.reshape(num_cells, -1)

    cells = np.arange(num_cells, dtype=np.int32)
    bs = form.function_spaces[0].dofmap.index_map_bs

    spline_loc_dofs = spline.getNumLocalDofs(block_size=bs)
    lagrange_loc_dofs = bs * (spline.getDegree() + 1) ** gdim

    if profile:
        perf_log.end_timing("Getting basic data")
        perf_log.start_timing("Creating dofmap")

    extraction_dofmap = spline.getExtractionOrdering(form.mesh)
    spline_dofmap = spline.getCpDofmap(extraction_dofmap, block_size=bs)

    if profile:
        perf_log.end_timing("Creating dofmap")

    if form.rank != 1 and form.rank != 2:
        raise ValueError("Ranks other than 1 and 2 are not supported")

    if profile:
        perf_log.start_timing("Packing constants")

    consts = dolfinx.cpp.fem.pack_constants(form._cpp_object)
    all_coeffs = dolfinx.cpp.fem.pack_coefficients(form._cpp_object)

    if profile:
        perf_log.end_timing("Packing constants")
        perf_log.start_timing("Computing extraction operators")

    extraction_operators = spline.get_lagrange_extraction_operators()
    tensor_product = spline.is_tensor_product_basis()

    perm = get_lagrange_permutation(
        form.function_spaces[0].element.basix_element.points, spline.getDegree(), gdim
    )
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
                    lagrange_loc_dofs,
                    spline_loc_dofs,
                    coeffs,
                    consts,
                    cells,
                    extraction_operators,
                    extraction_dofmap,
                    tensor_product,
                    permutation,
                    bs,
                    set_mat
                )

            elif form.rank == 1:
                _assemble_vector(
                    tensor.handle,
                    kernel,
                    vertices,
                    coords,
                    spline_dofmap,
                    lagrange_loc_dofs,
                    spline_loc_dofs,
                    coeffs,
                    consts,
                    cells,
                    extraction_operators,
                    extraction_dofmap,
                    tensor_product,
                    permutation,
                    bs,
                    set_vec
                )

            if profile:
                perf_log.end_timing(f"Assembling integral {i}")

    if profile:
        perf_log.end_timing("Assembly step")
        perf_log.end_timing(f"Assembling rank-{form.rank} form")


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _get_full_operator(
        operators: nb.typed.List[np.ndarray],
        bs: int,
        element: int,
        is_tensor_product: bool = False
) -> np.ndarray:
    """
    Assembles the full extraction matrix for a specified element. If the
    operators are tensor product, then the full operator is assembled by
    taking the Kronecker product of the operators. Otherwise, the operator
    is simply the operator corresponding to the element with a small hack
    that each operator is a 3D array with a single element in the first
    dimension.

    WARNING: numba will probably drop list support.

    Args:
        operators (list[np.ndarray]): list of operators
        bs (int): block size
        element (int): element index
        is_tensor_product (bool, optional): flag to indicate if the operators
            are tensor product. Default is False.

    Returns:
        np.ndarray: full extraction matrix with the correct block size
    """
    if is_tensor_product:
        if len(operators) == 1:
            return np.kron(
                operators[0][element], np.eye(bs, dtype=PETSc.ScalarType)
            )
        elif len(operators) == 2:
            j = element // operators[0].shape[0]
            i = element % operators[0].shape[0]

            return np.kron(
                np.kron(
                    operators[1][j],
                    operators[0][i]
                ),
                np.eye(bs, dtype=PETSc.ScalarType)
            )
        elif len(operators) == 3:
            k = element // (operators[0].shape[0] * operators[1].shape[0])
            j = (element // operators[0].shape[0]) % operators[1].shape[0]
            i = element % operators[0].shape[0]

            return np.kron(
                np.kron(
                    np.kron(
                        operators[2][k],
                        operators[1][j]
                    ),
                    operators[0][i]
                ),
                np.eye(bs, dtype=PETSc.ScalarType)
            )
    else:
        return np.kron(operators[element][0], np.eye(bs, dtype=PETSc.ScalarType))

@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _assemble_matrix(
        mat_handle: int,
        kernel: callable,
        vertices: np.ndarray,
        coords: np.ndarray,
        dofmap: np.ndarray | nb.typed.List[np.ndarray],
        lagrange_loc_dofs: int,
        spline_loc_dofs: np.ndarray,
        coeffs: np.ndarray,
        consts: np.ndarray,
        cells: np.ndarray,
        operators: nb.typed.List[np.ndarray],
        extraction_dofmap: np.ndarray,
        is_tensor_product: bool,
        permutation: np.ndarray,
        bs: int,
        set_vals: callable,
        mode: int = PETSc.InsertMode.ADD_VALUES
):
    """
    Matrix assembly kernel, inspired by the one presented in the DOLFINx
    paper. The list of arguments is a bit extensive to cover, so it is
    omitted here. The kernel is called for all cells which are passed in.
    Note that the handle of the matrix is passed in.
    """
    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    entity_local_index = np.array([0], dtype=np.intc)
    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    # Allocating memory for the local matrices
    lagrange_local = np.zeros(
        (lagrange_loc_dofs, lagrange_loc_dofs), dtype=PETSc.ScalarType
    )

    for cell in cells:
        # Assembling the full extraction operator
        element = extraction_dofmap[cell]
        full_kron = _get_full_operator(operators, bs, element, is_tensor_product)

        cell_coords[:, :] = coords[vertices[cell, :]]
        lagrange_local.fill(0.0)

        kernel(
            ffi.from_buffer(lagrange_local),
            ffi.from_buffer(coeffs[cell]),
            ffi.from_buffer(consts),
            ffi.from_buffer(cell_coords),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
        )

        # Permute rows and columns for index matching
        lagrange_local[:, :] = lagrange_local[permutation, :][:, permutation]
        spline_local = full_kron @ lagrange_local @ full_kron.T

        # If the number of dofs is constant, the first element contains it
        pos = dofmap[cell]
        n_dofs = spline_loc_dofs[0 if len(spline_loc_dofs) == 1 else cell]

        # PETSc insertion method, mode has to be a variable
        set_vals(
            mat_handle,
            n_dofs,
            pos.ctypes,
            n_dofs,
            pos.ctypes,
            spline_local.ctypes,
            mode
        )


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _assemble_vector(
        vec_handle: int,
        kernel: callable,
        vertices: np.ndarray,
        coords: np.ndarray,
        dofmap: np.ndarray | nb.typed.List[np.ndarray],
        lagrange_loc_dofs: int,
        spline_loc_dofs: np.ndarray,
        coeffs: np.ndarray,
        consts: np.ndarray,
        cells: np.ndarray,
        operators: nb.typed.List[np.ndarray],
        extraction_dofmap: np.ndarray,
        is_tensor_product: bool,
        permutation: np.ndarray,
        bs: int,
        set_vals: callable,
        mode: int = PETSc.InsertMode.ADD_VALUES
):
    """
    Vector assembly kernel, inspired by the one presented in the DOLFINx
    paper. The list of arguments is a bit extensive to cover, so it is
    omitted here. The kernel is called for all cells which are passed in.
    Note that a handle to the vector is passed in.
    """
    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    # Allocating memory for the local vectors
    lagrange_local = np.zeros(lagrange_loc_dofs, dtype=PETSc.ScalarType)

    for cell in cells:
        # Assembling the full extraction operator
        element = extraction_dofmap[cell]
        full_kron = _get_full_operator(operators, bs, element, is_tensor_product)

        cell_coords[:, :] = coords[vertices[cell, :]]
        lagrange_local.fill(0.0)

        kernel(
            ffi.from_buffer(lagrange_local),
            ffi.from_buffer(coeffs[cell]),
            ffi.from_buffer(consts),
            ffi.from_buffer(cell_coords),
            ffi.from_buffer(entity_local_index),
            ffi.from_buffer(perm),
        )

        spline_local = full_kron @ lagrange_local[permutation]

        # If the number of dofs is constant, the first element contains it
        pos = dofmap[cell]
        n_dofs = spline_loc_dofs[0 if len(spline_loc_dofs) == 1 else cell]

        # PETSc insertion method, mode has to be a variable
        set_vals(
            vec_handle,
            n_dofs,
            pos.ctypes,
            spline_local.ctypes,
            mode
        )


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _extract_control_points(
        cells: np.ndarray,
        spline_dofmap: np.ndarray,
        extraction_dofmap: np.ndarray,
        extraction_operators: nb.typed.List[np.ndarray],
        is_tensor_product: bool,
        fe_dofmap: np.ndarray,
        permutation: np.ndarray,
        control_points: np.ndarray,
        extracted_control_points: np.ndarray
):
    """
    This function extracts the control points from the spline basis to the
    Lagrange basis. It should be noted that for this function to work as
    intended one extra column is required in the control_points array.
    This column is used to store how many times a control point was repeatedly
    added to the global control point array.
    In the end one divides through the number of times a control point was
    added to get the correct control point values, but this is not part of
    this function!

    Args:
        cells (np.ndarray): cell indices for which to extract control points
        spline_dofmap (np.ndarray): spline dofmap
        extraction_dofmap (np.ndarray): extraction dofmap
        extraction_operators (list[np.ndarray]): extraction operators
        is_tensor_product (bool): flag to indicate if the operators
            are tensor product
        fe_dofmap (np.ndarray): finite element dofmap
        permutation (np.ndarray): permutation array
        control_points (np.ndarray): control points to extract from, the
            result is stored here, shape is (#cps, dim + 1)
        extracted_control_points (np.ndarray): extracted control points, the
            result is stored here, shape is (#extracted_cps, dim + 1)
    """
    for cell in cells:
        # Assembling the full extraction operator
        element = extraction_dofmap[cell]
        full_operator = _get_full_operator(
            extraction_operators, 1, element, is_tensor_product
        )

        # Determining corresponding local dof ranges
        local_cp_range = spline_dofmap[cell]
        local_fe_range = fe_dofmap[cell][permutation]

        # Simple insertion because of numpy
        extracted_control_points[local_fe_range, :] += (
            full_operator.T @ (control_points[local_cp_range, :])
        )
