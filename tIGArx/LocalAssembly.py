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

    spline_loc_dofs = spline.getNumLocalDofs(block_size=bs)
    lagrange_loc_dofs = bs * (spline.getDegree() + 1) ** gdim

    if profile:
        perf_log.end_timing("Getting basic data")
        perf_log.start_timing("Creating dofmap")

    extraction_dofmap = spline.getExtractionOrdering(form.mesh)
    spline_dofmap = spline.getCpDofmap(extraction_dofmap, block_size=bs)

    if profile:
        perf_log.end_timing("Creating dofmap")

    # The object that is passed to the assembly routines
    tensor: PETSc.Mat | PETSc.Vec

    dimension = spline.getNcp() * bs

    if form.rank == 2:
        if profile:
            perf_log.start_timing("Computing pre-allocation")

        csr_allocation = spline.getCSRPrealloc(block_size=bs)

        if profile:
            perf_log.end_timing("Computing pre-allocation")
            perf_log.start_timing("Allocating rank-2 tensor")

        tensor = PETSc.Mat(form.mesh.comm)
        tensor.createAIJ(dimension, dimension, csr=csr_allocation)

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
                    lagrange_loc_dofs,
                    spline_loc_dofs,
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
def get_full_operator(operators, bs, gdim, element):
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
        mat_handle,
        kernel,
        vertices,
        coords,
        dofmap,
        lagrange_loc_dofs,
        spline_loc_dofs,
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

    # Allocating memory for the local matrices
    lagrange_local = np.zeros(
        (lagrange_loc_dofs, lagrange_loc_dofs), dtype=PETSc.ScalarType
    )

    for cell in cells:
        element = extraction_dofmap[cell]
        full_kron = get_full_operator(operators, bs, gdim, element)

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

        # Permute the rows and columns so that they match the
        # indexing of the spline basis
        lagrange_local[:, :] = lagrange_local[permutation, :][:, permutation]
        spline_local = full_kron @ lagrange_local @ full_kron.T

        pos = dofmap[cell]
        n_dofs = spline_loc_dofs[0] if len(spline_loc_dofs) == 1 else spline_loc_dofs[cell]

        # mat_handle.setValues(pos, pos, spline_local, PETSc.InsertMode.ADD_VALUES)
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
        vec,
        kernel,
        vertices,
        coords,
        dofmap,
        lagrange_loc_dofs,
        spline_loc_dofs,
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

    # Allocating memory for the local vectors
    lagrange_local = np.zeros(lagrange_loc_dofs, dtype=PETSc.ScalarType)

    for cell in cells:
        element = extraction_dofmap[cell]
        full_kron = get_full_operator(operators, bs, gdim, element)

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

        pos = dofmap[cell]
        n_dofs = spline_loc_dofs[0] if len(spline_loc_dofs) == 1 else spline_loc_dofs[cell]

        set_vals(
            vec,
            n_dofs,
            pos.ctypes,
            spline_local.ctypes,
            mode
        )


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _extract_control_points(
        cells,
        spline_dofmap,
        extraction_dofmap,
        extraction_operators,
        fe_dofmap,
        permutation,
        control_points,
        extracted_control_points,
        space_dim,
):
    for cell in cells:
        element = extraction_dofmap[cell]
        full_operator = get_full_operator(extraction_operators, 1, space_dim, element)

        local_cp_range = spline_dofmap[cell]
        local_fe_range = fe_dofmap[cell][permutation]

        extracted_control_points[local_fe_range, :] += (
            full_operator.T @ (control_points[local_cp_range, :])
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
