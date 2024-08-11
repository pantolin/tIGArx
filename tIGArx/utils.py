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



