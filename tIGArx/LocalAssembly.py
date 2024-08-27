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
def get_full_operator(operators, bs, gdim, element):
    if gdim == 1 or len(operators) == 1:
        return np.kron(operators[0][element], np.eye(bs, dtype=PETSc.ScalarType))

    elif gdim == 2:
        j = element // operators[0].shape[0]
        i = element % operators[0].shape[0]

        return np.kron(
            np.kron(
                operators[1][j],
                operators[0][i]
            ),
            np.eye(bs, dtype=PETSc.ScalarType)
        )
    else:
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

    for cell in cells:
        element = extraction_dofmap[cell]
        full_kron[:, :] = get_full_operator(operators, bs, gdim, element)

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

    for cell in cells:
        element = extraction_dofmap[cell]
        full_kron[:, :] = get_full_operator(operators, bs, gdim, element)

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
