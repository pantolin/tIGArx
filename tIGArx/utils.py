import numpy as np
import numba as nb

import dolfinx
import basix
import basix.ufl
import petsc4py.PETSc as PETSc

from tIGArx.common import worldcomm, INDEX_TYPE, mpisize, mpirank


def generateMeshXMLFileName(comm):
    import hashlib

    s = repr(comm) + repr(comm.Get_rank())
    return "mesh-" + str(hashlib.md5(s.encode("utf-8")).hexdigest()) + ".xml"


# helper function to do MatMultTranspose() without all the setup steps for the
# results vector
def multTranspose(M, b):
    """
    Returns ``M^T*b``, where ``M`` and ``b`` are PETSc matrix and tensor
    objects.
    """
    MTb = M.createVecRight()
    M.multTranspose(b, MTb)
    MTb.assemble()

    return MTb


# helper function to generate an identity permutation IS
# given an ownership range
def generateIdentityPermutation(ownRange, comm=worldcomm):
    """
    Returns a PETSc index set corresponding to the ownership range.
    """

    iStart = ownRange[0]
    iEnd = ownRange[1]
    localSize = iEnd - iStart
    iArray = np.zeros(localSize, dtype=INDEX_TYPE)
    for i in np.arange(0, localSize):
        iArray[i] = i + iStart
    # FIXME to simplify as iArray = np.arange(ownRange[0], ownRange[1], dtype=INDEX_TYPE)
    retval = PETSc.IS(comm)
    retval.createGeneral(iArray, comm=comm)
    return retval


def getCellType(dim):
    assert 1 <= dim <= 3, "Invalid dimension."

    if dim == 1:
        cell = basix.CellType.interval
    elif dim == 2:
        cell = basix.CellType.quadrilateral
    else:
        cell = basix.CellType.hexahedron
    return cell


def createElementType(degree, dim, discontinuous):
    """
    Returns an UFL element of the given degree either continuous or
    discontinuous, depending on the return of useDG().
    #FIXME to document better.
    """

    cell = getCellType(dim)
    family = basix.ElementFamily.P
    variant = basix.LagrangeVariant.equispaced

    ufl_elem = basix.ufl.element(
        family=family, cell=cell, degree=degree,
        lagrange_variant=variant, discontinuous=discontinuous
    )
    return ufl_elem


def create_permuted_element(degree: int , dim: int, dofs_per_cp=1, discontinuous=False):
    """
    Returns an UFL element of the given degree either continuous or
    discontinuous, although the discontinuous feature should not be
    used in general. The DOFs of this element are permuted so that
    they are ordered in a "lexicographic" way, or Kronecker product
    way, which is the way that the DOFs are ordered in spline
    spaces. This means a traversal like the following 3rd degree
    2D element (x-axis to right, y-axis up):
    12 -- 13 -- 14 -- 15
    |     |     |     |
    8 --- 9 --- 10 -- 11
    |     |     |     |
    4 --- 5 --- 6 --- 7
    |     |     |     |
    0 --- 1 --- 2 --- 3

    Regular dof ordering would be:
    2 --- 10 -- 11 -- 3
    |     |     |     |
    7 --- 14 -- 15 -- 9
    |     |     |     |
    6 --- 13 -- 12 -- 8
    |     |     |     |
    0 --- 4 --- 5 --- 1
    Note: this is an active issue and check whether it was resolved
    https://github.com/FEniCS/basix/issues/846,

    Args:
        degree (int): degree of the element
        dim (int): dimension of the element
        dofs_per_cp (int): number of degrees of freedom per control point
        discontinuous (bool): whether the element is discontinuous
    """

    base_element = basix.create_element(
        basix.ElementFamily.P,
        getCellType(dim),
        degree,
        basix.LagrangeVariant.equispaced,
        basix.DPCVariant.unset,
        discontinuous
    )

    perm = get_lagrange_permutation(base_element.points, degree, dim)

    element = basix.create_element(
        basix.ElementFamily.P,
        getCellType(dim),
        degree,
        basix.LagrangeVariant.equispaced,
        basix.DPCVariant.unset,
        discontinuous,
        perm.tolist()
    )

    ufl_element = basix.ufl._BasixElement(element)
    if dofs_per_cp > 1:
        ufl_element = basix.ufl.blocked_element(ufl_element, shape=(dofs_per_cp,))

    return ufl_element


def createVectorElementType(degrees, dim, discontinuous, nFields):
    """
    Returns an UFL element of the given degree either continuous or
    discontinuous, depending on the return of useDG().
    #FIXME to document better.
    """

    assert len(degrees) == nFields

    if len(degrees) == 1:
        ufl_elem = createElementType(degrees[0], dim, discontinuous)

    elif len(set(degrees)) == 1:  # Isotropic degrees

        scalar_elem = createElementType(degrees[0], dim, discontinuous)

        ufl_elem = basix.ufl.element(
            family=scalar_elem.family_name,
            cell=scalar_elem.cell_type,
            degree=scalar_elem.element.degree,
            lagrange_variant=scalar_elem.lagrange_variant,
            discontinuous=scalar_elem.discontinuous,
            shape=(nFields,)
        )
    else:
        if not isDolfinxVersion8orHigher() and mpisize > 1:
            raise Exception("Currently, due to a dolfinx bug in versions < 0.8.0, "
                            "it is not possible to use mixed element in parallel. "
                            "See https://fenicsproject.discourse.group/t/degrees-of-freedom-of-sub-spaces-in-parallel/14351")

        scalar_elems = [createElementType(
            deg, dim, discontinuous) for deg in degrees]
        ufl_elem = basix.ufl.mixed_element(scalar_elems)

    return ufl_elem


def isDolfinxVersion8orHigher():
    """ FIXME to document
    """
    from packaging.version import Version
    return Version(dolfinx.__version__) >= Version("0.8.0")


def createFunctionSpace(mesh, ufl_elem):
    """ FIXME to document
    """
    if isDolfinxVersion8orHigher():
        V = dolfinx.fem.functionspace(mesh, ufl_elem)
    else:
        V = dolfinx.fem.FunctionSpace(mesh, ufl_elem)
    return V


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
def interleave_and_expand_numba(arr: np.ndarray, n: int) -> np.ndarray:
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
    interleaved = np.zeros(arr.size * n, dtype=np.int32)
    for d in range(arr.size):
        for b in range(n):
            interleaved[d * n + b] = np.int32(arr[d] * n + b)

    return interleaved


def get_lagrange_permutation(dof_coords: np.ndarray, deg: int, gdim: int):
    """
    Get permutation for Lagrange basis

    Args:
        dof_coords (np.array): dof coordinates for element
        deg (int): degree of the basis
        gdim (int): geometric dimension
    Returns:
        permutation (np.array): permutation array
    """
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


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def get_csr_pre_allocation(cells, dofmap, rows, max_dofs_per_row):
    """
    Quite an inefficient way to get the preallocation of a CSR matrix
    from a dofmap. While it is hardly ideal, it is not terrible for
    problems with 2D surface meshes, while it is lacking for 3D meshes.
    It is reintroduced here to help with T-splines, where the dofmap
    seems to be the most natural way to get the sparsity pattern.

    Args:
        cells (np.ndarray): array of cells
        dofmap (np.ndarray): dofmap
        rows (int): number of rows, i.e. number of dofs (or cps)
        max_dofs_per_row (int): maximum number of dofs per row, maximum
            number of interacting cps per each cp

    Returns:
        index_ptr (np.ndarray): array of pointers to the indices array
        indices (np.ndarray): array of indices
    """
    dofs_per_row = np.zeros((rows, max_dofs_per_row), dtype=np.int32)
    nnz_per_row = np.zeros(rows, dtype=np.int32)

    for cell in cells:
        for row_idx in dofmap[cell]:
            for dof in dofmap[cell]:
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
