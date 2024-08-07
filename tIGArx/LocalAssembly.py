import numpy as np
import numba as nb

from cffi import FFI
from petsc4py import PETSc

from tIGArx.SplineInterface import AbstractScalarBasis

ffi = FFI()

import dolfinx


def assemble_matrix(form, spline: AbstractScalarBasis):
    """
    Assemble matrix

    Args:
        form (dolfinx.fem.Form): form to assemble

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

    mat = PETSc.Mat(form.mesh.comm)
    mat.createAIJ(spline.getNcp(), spline.getNcp())

    max_nonzeros_per_row = 0
    row_nnz = [0] * spline.getNcp()
    for dofs in spline_dofmap:
        for dof in dofs:
            row_nnz[dof] += 1

    mat.setPreallocationNNZ(row_nnz)

    return mat


def _assemble_cells(
    Ah,
    kernel,
    vertices,
    coords,
    dofmap,
    num_loc_dofs,
    coeffs,
    consts,
    cells,
    set_vals,
    mode,
):
    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    A_local = np.zeros((num_loc_dofs, num_loc_dofs), dtype=PETSc.ScalarType)
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    for k, cell in enumerate(cells):
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

        set_vals(
            Ah,
            num_loc_dofs,
            ffi.from_buffer(pos),
            num_loc_dofs,
            ffi.from_buffer(pos),
            ffi.from_buffer(A_local),
            mode,
        )

@nb.njit
def assemble_cells(
    Ah,
    kernel,
    vertices,
    coords,
    dofmap,
    num_loc_dofs,
    coeffs,
    consts,
    cells,
    set_vals,
    mode,
):
    # Initialize
    num_loc_vertices = vertices.shape[1]
    cell_coords = np.zeros((num_loc_vertices, 3))
    A_local = np.zeros((num_loc_dofs, num_loc_dofs), dtype=PETSc.ScalarType)
    entity_local_index = np.array([0], dtype=np.intc)

    # Don't permute
    perm = np.array([0], dtype=np.uint8)

    for k, cell in enumerate(cells):
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
        # print the local matrix for debugging
        # print("Local matrix for cell ", k)
        # print(A_local)

        set_vals(
            Ah,
            num_loc_dofs,
            ffi.from_buffer(pos),
            num_loc_dofs,
            ffi.from_buffer(pos),
            ffi.from_buffer(A_local),
            mode,
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
