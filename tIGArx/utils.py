import numpy as np

import dolfinx
import basix
import petsc4py.PETSc as PETSc

from mpi4py import MPI

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
