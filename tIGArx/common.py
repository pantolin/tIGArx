"""
The ``common`` module
---------------------
contains basic definitions of abstractions for
generating extraction data and importing it again for use in analysis.  Upon
importing this module, a number of setup steps are carried out
(e.g., initializing MPI).
"""

import ufl.equation
import scipy as sp
import numpy as np
import abc
from mpi4py import MPI
from petsc4py import PETSc
from tIGArx.calculusUtils import (
    getMetric,
    mappedNormal,
    tIGArxMeasure,
    volumeJacobian,
    surfaceJacobian,
    cartesianGrad,
    cartesianDiv,
    cartesianCurl,
    pinvD,
    getChristoffel,
    CurvilinearTensor,
    curvilinearGrad,
    curvilinearDiv,
)

import dolfinx
from dolfinx.io import XDMFFile
from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector
from dolfinx import default_real_type
import basix
import petsc4py
import sys


petsc4py.init(sys.argv)


worldcomm = MPI.COMM_WORLD
selfcomm = MPI.COMM_SELF

mpisize = worldcomm.Get_size()
mpirank = worldcomm.Get_rank()


# FIXME to figure out right indices.
INDEX_TYPE = "int32"
# DEFAULT_PREALLOC = 100
DEFAULT_PREALLOC = 500

# Choose default behavior for permutation of indices based on the number
# of MPI tasks
if mpisize > 8:
    DEFAULT_DO_PERMUTATION = True
else:
    DEFAULT_DO_PERMUTATION = False

# basis function evaluations less than this will be considered outside the
# function's support
DEFAULT_BASIS_FUNC_IGNORE_EPS = 10.0 * np.finfo(default_real_type).eps

# DEFAULT_LINSOLVER_REL_TOL = 10.0 * np.finfo(default_real_type).eps
DEFAULT_LINSOLVER_REL_TOL = 1.0e-10
DEFAULT_LINSOLVER_ABS_TOL = PETSc.DEFAULT
DEFAULT_LINSOLVER_MAX_ITERS = 1000

# This was too small for optimal convergence rates in high-order biharmonic
# discretizations with highly-refined meshes:
# DEFAULT_BASIS_FUNC_IGNORE_EPS = 1e-9


# file naming conventions
EXTRACTION_MESH_FILE = "extraction-mesh.xdmf"
EXTRACTION_INFO_FILE = "extraction-info.txt"

# FIXME is this really needed ?


def EXTRACTION_H5_CONTROL_FUNC_NAME(dim):
    return "/control" + str(dim)


EXTRACTION_ZERO_DOFS_FILE = "zero-dofs.h5"
EXTRACTION_MAT_FILE = "extraction-mat.h5"
EXTRACTION_MAT_FILE_CTRL = "extraction-mat-ctrl.h5"
EXTRACTION_VEC_FILE_CTRL_PTS = "extraction-vec-ctrl-pts.h5"

# DG space is more memory-hungry, but allows for $C^{-1}$-continuous splines,
# e.g., for div-conforming VMS, and will still work for more continuous
# spaces.
USE_DG_DEFAULT = True

# whether or not to explicitly form M^T (memory vs. speed tradeoff)
FORM_MT = False

# Helper function to generate unique temporary file names for dolfinx
# XML meshes; file name is unique for a given rank on a given communicator.


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
    assert 1 <= dim and dim <= 3, "Invalid dimension."

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
        family=family, cell=cell, degree=degree, lagrange_variant=variant, discontinuous=discontinuous)
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
            family=scalar_elem.family_name, cell=scalar_elem.cell_type, degree=scalar_elem.element.degree, lagrange_variant=scalar_elem.lagrange_variant, discontinuous=scalar_elem.discontinuous, shape=(nFields,))
    else:
        if not isDolfinxVersion8orHigher() and mpisize > 1:
            raise Exception("Currently, due to a dolfinx bug in versions < 0.8.0, it is not possible to use mixed element in parallel. See https://fenicsproject.discourse.group/t/degrees-of-freedom-of-sub-spaces-in-parallel/14351")

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


class AbstractExtractionGenerator(object):
    """
    Abstract class representing the minimal set of functions needed to write
    extraction operators for a spline.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, comm, *args):
        """
        Arguments in ``*args`` are passed as a tuple to
        ``self.customSetup()``.  Appropriate arguments vary by subclass.  If
        the first argument ``comm`` is of type ``petsc4py.PETSc.Comm``, then
        it will be treated as a communicator for the extraction generator.
        Otherwise, it is treated as if it were the first argument in ``args``.
        """

        if not (type(comm) == type(selfcomm)):
            args = (comm,) + args
            self.comm = worldcomm
        else:
            self.comm = comm

        self.customSetup(args)
        self.genericSetup()

    def getComm(self):
        """
        Returns the extraction generator's MPI communicator.
        """
        return self.comm

    # what type of element (CG or DG) to extract to
    # (override in subclass for non-default behavior)
    def useDG(self):
        """
        Returns a Boolean, indicating whether or not to use DG elements
        in extraction.
        """

        return USE_DG_DEFAULT

    def extractionElement(self):
        """
        Returns a string indicating what type of FE to use in extraction.
        """

        if self.useDG():
            return "DG"
        else:
            return "P"

    def createElementType(self, degree):
        """
        Returns an UFL element of the given degree either continuous or
        discontinuous, depending on the return of useDG().
        # FIXME to document better
        """

        dim = self.mesh.topology.dim
        discontinuous = self.useDG()
        ufl_elem = createElementType(degree, dim, discontinuous)
        return ufl_elem

    def createVectorElementType(self, degrees, nFields):
        """
        Returns an UFL vector/mixed element of the given degrees either continuous or
        discontinuous, depending on the return of useDG().
        # FIXME to document better
        """

        if len(degrees) == 1:
            ufl_elem = self.createElementType(degrees[0])
        else:
            dim = self.mesh.topology.dim
            discontinuous = self.useDG()
            ufl_elem = createVectorElementType(
                degrees, dim, discontinuous, nFields)
        return ufl_elem

    @abc.abstractmethod
    def customSetup(self, args):
        """
        Customized instructions to execute during initialization.  ``args``
        is a tuple of custom arguments.
        """
        return

    @abc.abstractmethod
    def getNFields(self):
        """
        Returns the number of unknown fields for the spline.
        """
        return

    @abc.abstractmethod
    def getHomogeneousCoordinate(self, node, direction):
        """
        Return the ``direction``-th homogeneous coordinate of the ``node``-th
        control point of the spline.
        """
        return

    @abc.abstractmethod
    def generateMesh(self):
        """
        Generate and return an FE mesh suitable for extracting the
        subclass's spline space.
        """
        return

    @abc.abstractmethod
    def getDegree(self, field):
        """
        Return the degree of polynomial to be used in the extracted
        representation of a given ``field``, with ``-1`` being the
        control field.
        """
        return

    @abc.abstractmethod
    def getNcp(self, field):
        """
        Return the total number of degrees of freedom of a given ``field``,
        with field ``-1`` being the control mesh field.
        """
        return

    @abc.abstractmethod
    def getNsd(self):
        """
        Return the number of spatial dimensions of the physical domain.
        """
        return

    def globalDof(self, field, localDof):
        """
        Given a ``field`` and a local DoF number ``localDof``,
        return the global DoF number;
        this is BEFORE any re-ordering for parallelization.
        """
        # offset localDof by
        retval = localDof
        for i in range(0, field):
            retval += self.getNcp(i)
        return retval

    def generatePermutation(self):
        """
        Generates an index set to permute the IGA degrees of freedom
        into an order that is (hopefully) efficient given the partitioning
        of the FEM nodes.  Assume that ``self.M`` currently holds the
        un-permuted extraction matrix.
        Default implementation just fills in an identity permutation.
        """
        return generateIdentityPermutation(
            self.M.getOwnershipRangeColumn(), self.comm
        )

    def addZeroDofsGlobal(self, newDofs):
        """
        Adds new DoFs in the list ``newDofs`` in global numbering
        to the list of DoFs to which
        homogeneous Dirichlet BCs will be applied during analysis.
        """
        self.zeroDofs += newDofs

    def addZeroDofs(self, field, newDofs):
        """
        Adds new DoFs in the list ``newDofs`` in local numbering for a
        given ``field`` to the list of DoFs to which
        homogeneous Dirichlet BCs will be applied during analysis.
        """
        # FIXME maybe just newDofsGlobal = np.empty_like(newDofs)
        newDofsGlobal = newDofs[:]
        for i in range(0, len(newDofs)):
            newDofsGlobal[i] = self.globalDof(field, newDofs[i])
        self.addZeroDofsGlobal(newDofsGlobal)

    def getPrealloc(self, control):
        """
        Returns the number of entries per row needed in the extraction matrix.
        The parameter ``control`` is a Boolean indicating whether or not this
        is the preallocation for the scalar field used for control point
        coordinates.

        If left as the default, this could potentially slow down drastically
        for very high-order splines, or waste a lot of memory for low order
        splines.  In general, it is a good idea to override this in
        subclasses.
        """
        return DEFAULT_PREALLOC

    def getIgnoreEps(self):
        """
        Returns an absolute value below which basis function evaluations are
        considered to be outside of the function's support.

        This method is very unlikely to require overriding in subclasses.
        """
        return DEFAULT_BASIS_FUNC_IGNORE_EPS

    @abc.abstractmethod
    def generateM_control(self):
        """
        Return the extraction matrix for the control field.
        """
        return

    @abc.abstractmethod
    def generateM(self):
        """
        Return the extraction matrix for the unknowns.
        """
        return

    def genericSetup(self):
        """
        Common setup steps for all subclasses (called in ``self.__init__()``).
        """

        self.mesh = self.generateMesh()

        # note: if self.nsd is set in a customSetup, then the subclass
        # getNsd() references that, this is still safe
        self.nsd = self.getNsd()

        self.VE_control = self.createElementType(self.getDegree(-1))
        self.V_control = createFunctionSpace(self.mesh, self.VE_control)

        nFields = self.getNFields()
        degrees = [self.getDegree(i) for i in range(nFields)]
        self.VE = self.createVectorElementType(degrees, nFields)
        self.V = createFunctionSpace(self.mesh, self.VE)

        self.cpFuncs = [dolfinx.fem.Function(
            self.V_control) for _ in range(self.nsd + 1)]

        self.M_control = self.generateM_control()
        self.M = self.generateM()

        # get transpose
        if FORM_MT:
            MT_control = self.M_control.transpose(PETSc.Mat(self.comm))

        # generating CPs, weights in spline space:
        # (control net never permuted)
        for i in range(0, self.nsd + 1):
            # FIXME why computing product for later on setting coordinates manually?
            if FORM_MT:
                MTC = MT_control * self.cpFuncs[i].vector
            else:
                MTC = multTranspose(self.M_control, self.cpFuncs[i].vector)
            for I in np.arange(*MTC.getOwnershipRange()):
                MTC[I] = self.getHomogeneousCoordinate(I, i)
            MTC.assemble()

            MTCM = self.M_control * MTC

            size_local = self.cpFuncs[i].x.index_map.size_local
            self.cpFuncs[i].x.array[:size_local] = MTCM.array_r
            self.cpFuncs[i].x.scatter_forward()

        # may need to be permuted
        self.zeroDofs = []  # self.generateZeroDofs()

        # replace M with permuted version
        # if(mpisize > 1):
        #
        #    self.permutation = self.generatePermutation()
        #    id_permutation = generateIdentityPermutation(self.M.getOwnershipRange())
        #    self.M = self.M.permute(id_permutation, self.permutation)
        #
        #    # fix list of zero DOFs
        #    self.permutationAO = PETSc.AO()
        #    id_permutation_col = generateIdentityPermutation(self.M.getOwnershipRangeColumn())
        #    self.permutationAO.createBasic(self.permutation, id_permutation_col)
        #    zeroDofIS = PETSc.IS()
        #    zeroDofIS.createGeneral(np.array(self.zeroDofs,dtype=INDEX_TYPE))
        #    self.zeroDofs = self.permutationAO.app2petsc(zeroDofIS).getIndices()

    def applyPermutation(self):
        """
        Permutes the order of the IGA degrees of freedom, so that their
        parallel partitioning better aligns with that of the FE degrees
        of freedom, which is generated by standard mesh-partitioning
        approaches in FEniCS.
        """
        if self.comm.Get_size() > 1:

            self.permutation = self.generatePermutation()
            id_perm = generateIdentityPermutation(
                self.M.getOwnershipRange(), self.comm)
            self.M = self.M.permute(id_perm, self.permutation)

            # fix list of zero DOFs
            self.permutationAO = PETSc.AO(self.comm)

            id_perm_col = generateIdentityPermutation(
                self.M.getOwnershipRangeColumn(), self.comm)
            self.permutationAO.createBasic(self.permutation, id_perm_col)
            zeroDofIS = PETSc.IS(self.comm)
            zeroDofIS.createGeneral(np.array(self.zeroDofs, dtype=INDEX_TYPE))
            self.zeroDofs = self.permutationAO.app2petsc(
                zeroDofIS).getIndices()

    def writeExtraction(self, dirname, doPermutation=DEFAULT_DO_PERMUTATION):
        """
        Writes all extraction data to files in a directory named
        ``dirname``.  The optional argument ``doPermutation`` is a Boolean
        indicating whether or not to permute the unknowns for better
        parallel performance in matrix--matrix multiplications.  (Computing
        this permuation may be slow for large meshes.)
        """
        # need:
        # - HDF5 file w/
        # -- mesh
        # -- extracted CPs, weights
        # - Serialized PETSc matrix for M_control
        # - Serialized PETSc matrix for M
        # - txt file w/
        # -- nsd
        # -- number of fields
        # -- for each field (+ scalar control field)
        # --- function space info (element type, degree)
        # - File for each processor listing zero-ed dofs

        assert False, "Functionality not tested yet"

        if doPermutation:
            self.applyPermutation()

        # write mesh XDMF file
        with dolfinx.io.XDMFFile(self.mesh.comm, dirname + "/" + EXTRACTION_MESH_FILE, "w") as xdmf:
            xdmf.write_mesh(self.mesh)

        # PETSc matrices

        viewer = PETSc.ViewerHDF5(self.comm).createBinary(
            dirname + "/" + EXTRACTION_VEC_FILE_CTRL_PTS, "w"
        )
        for i in range(0, self.nsd + 1):
            self.generateMesh()
            viewer.view(self.cpFuncs[i].vector)

        viewer = PETSc.ViewerHDF5(self.comm).createBinary(
            dirname + "/" + EXTRACTION_MAT_FILE, "w"
        )
        viewer(self.M)
        viewer = PETSc.ViewerHDF5(self.comm).createBinary(
            dirname + "/" + EXTRACTION_MAT_FILE_CTRL, "w"
        )
        viewer(self.M_control)

        # write out zero-ed dofs
        # dofList = self.zeroDofs
        # fs = ""
        # for dof in dofList:
        #    fs += str(dof)+" "
        # f = open(dirname+"/"+EXTRACTION_ZERO_DOFS_FILE,"w")
        # f.write(fs)
        # f.close()
        zeroDofIS = PETSc.IS(self.comm)
        zeroDofIS.createGeneral(np.array(self.zeroDofs, dtype=INDEX_TYPE))
        viewer = PETSc.ViewerHDF5(self.comm).createBinary(
            dirname + "/" + EXTRACTION_ZERO_DOFS_FILE, "w"
        )
        viewer(zeroDofIS)

        # write info
        if mpirank == 0:
            fs = (
                str(self.nsd)
                + "\n"
                + self.extractionElement()
                + "\n"
                + str(self.getNFields())
                + "\n"
            )
            for i in range(-1, self.getNFields()):
                fs += str(self.getDegree(i)) + "\n" + \
                    str(self.getNcp(i)) + "\n"
            f = open(dirname + "/" + EXTRACTION_INFO_FILE, "w")
            f.write(fs)
            f.close()
        self.comm.Barrier()


# class ExtractedNonlinearProblem(NonlinearProblem):
#     """
#     Class encapsulating a nonlinear problem posed on an extracted spline, to
#     allow existing nonlinear solvers (e.g., PETSc SNES) to be used.

#     NOTE: Obtaining the initial guess for the IGA DoFs from the given
#     FE function for the solution fields currently requires
#     a linear solve, which is performed using the spline object's solver,
#     if any.
#     """

#     def __init__(self, spline, residual, tangent, solution, **kwargs):
#         """
#         The argument ``spline`` is an ``ExtractedSpline`` on which the
#         problem is solved.  ``residual`` is the residual form of the problem.
#         ``tangent`` is the Jacobian of this form.  ``solution`` is a
#         ``Function`` in ``spline.V``.  Additional keyword arguments will be
#         passed to the superclass constructor.
#         """
#         super(ExtractedNonlinearProblem, self).__init__(**kwargs)
#         self.spline = spline
#         self.solution = solution
#         self.residual = residual
#         self.tangent = tangent

#     # Override methods from NonlinearProblem to perform extraction:
#     def form(self, A, P, B, x):
#         self.solution.vector()[:] = self.spline.M * x

#     def F(self, b, x):
#         b[:] = self.spline.assembleVector(self.residual)
#         return b

#     def J(self, A, x):
#         M = self.spline.assembleMatrix(self.tangent).mat()
#         A.mat().setSizes(M.getSizes())
#         A.mat().setUp()
#         A.mat().assemble()
#         M.copy(result=A.mat())
#         return A


class ExtractedNonlinearSolver:
    """
    Class encapsulating the extra work surrounding a nonlinear solve when
    the problem is posed on an ``ExtractedSpline``.
    """

    def __init__(self, problem, solver):
        """
        ``problem`` is an ``ExtractedNonlinearProblem``, while ``solver``
        is either a ``NewtonSolver`` or a ``PETScSNESSolver``
        that will be used behind the scenes.
        """
        self.problem = problem
        self.solver = solver

    def solve(self):
        """
        This method solves ``self.problem``, using ``self.solver`` and updating
        ``self.problem.solution`` with the solution (in extracted FE
        representation).
        """

        # Need to solve a linear problem for initial guess for IGA DoFs; any
        # way around this?
        tempVec = self.problem.spline.FEtoIGA(self.problem.solution)

        # tempFunc = Function(self.problem.spline.V)
        # tempFunc.assign(self.problem.solution)
        # RHS of problem for initial guess IGA DoFs:
        # MTtemp = self.problem.spline.extractVector(tempFunc.vector(),
        #                                           applyBCs=False)
        # Vector with right dimension for IGA DoFs (content doesn't matter):
        # tempVec = self.problem.spline.extractVector(tempFunc.vector())
        # LHS of problem for initial guess:
        # Mm = as_backend_type(self.problem.spline.M).mat()
        # MTMm = Mm.transposeMatMult(Mm)
        # MTM = PETScMatrix(MTMm)
        # if(self.problem.spline.linearSolver == None):
        #    solve(MTM,tempVec,MTtemp)
        # else:
        #    self.problem.spline.linearSolver.solve(MTM,tempVec,MTtemp)
        self.solver.solve(self.problem, tempVec)

        self.problem.solution.vector()[:] = self.problem.spline.M * tempVec


# class SplineDisplacementExpression(Expression):
#
#    """
#    An expression that can be used to evaluate ``F`` plus an optional
#    displacement at arbitrary points.  To be usable, it must have the
#    following attributes assigned:
#
#    (1) ``self.spline``: an instance of ``ExtractedSpline`` to which the
#    displacement applies.
#
#    (2) ``self.functionList:`` a list of scalar functions in the
#    function space for ``spline``'s control mesh, which act as components of
#    the displacement. If ``functionList`` contains too few entries (including
#    zero entries), the missing entries are assumed to be zero.
#    """
#
#    # needs attributes:
#    # - spline (ExtractedSpline)
#    # - functionList (list of SCALAR Functions)
#
#    def eval_cell(self,values,x,c):
#        phi = []
#        out = array([0.0,])
#        for i in range(0,self.spline.nsd):
#            self.spline.cpFuncs[i].set_allow_extrapolation(True)
#            #phi += [self.cpFuncs[i](Point(x)),]
#            self.spline.cpFuncs[i].eval_cell(out,x,c)
#            phi += [out[0],]
#        self.spline.cpFuncs[self.spline.nsd].set_allow_extrapolation(True)
#        for i in range(0,self.spline.nsd):
#            if(i<len(self.functionList)):
#                self.functionList[i].set_allow_extrapolation(True)
#                self.functionList[i].eval_cell(out,x,c)
#                phi[i] += out[0]
#        #w = self.cpFuncs[self.nsd](Point(x))
#        self.spline.cpFuncs[self.spline.nsd].eval_cell(out,x,c)
#        w = out[0]
#        for i in range(0,self.spline.nsd):
#            phi[i] = phi[i]/w
#        xx = []
#        for i in range(0,self.spline.nsd):
#            if(i<len(x)):
#                xx += [x[i],]
#            else:
#                xx += [0,]
#        for i in range(0,self.spline.nsd):
#            values[i] = phi[i] - xx[i]
#
#    #def value_shape(self):
#    #    return (self.spline.nsd,)


# compose with deformation
# class tIGArxExpression(Expression):
#
#    """
#    A subclass of ``Expression`` which composes its attribute ``self.expr``
#    (also an ``Expression``) with the deformation ``F`` given by its attribute
#    ``self.cpFuncs``, which is a list of ``Function`` objects, specifying the
#    components of ``F``.
#    """
#
#    # using eval_cell allows us to avoid having to search for which cell
#    # x is in; also x need not be in a unique cell, which is nice for
#    # splines that do not have a single coordinate chart
#    def eval_cell(self,values,x,c):
#        phi = []
#        out = array([0.0,])
#        for i in range(0,self.nsd):
#            self.cpFuncs[i].set_allow_extrapolation(True)
#            self.cpFuncs[i].eval_cell(out,x,c)
#            phi += [out[0],]
#        self.cpFuncs[self.nsd].set_allow_extrapolation(True)
#        self.cpFuncs[self.nsd].eval_cell(out,x,c)
#        w = out[0]
#        for i in range(0,self.nsd):
#            phi[i] = phi[i]/w
#        self.expr.eval(values,array(phi))


# could represent any sort of spline that is extractable
class ExtractedSpline(object):
    """
    A class representing an extracted spline.  The idea is that all splines
    look the same after extraction, so there is no need for a proliferation
    of different classes to cover NURBS, T-splines, etc. (as there is for
    extraction generators).
    """

    def __init__(
        self,
        sourceArg,
        quadDeg,
        mesh=None,
        doPermutation=DEFAULT_DO_PERMUTATION,
        comm=worldcomm,
    ):
        """
        Generates instance from extraction data in ``sourceArg``, which
        might either be an ``AbstractExtractionGenerator`` or the name of
        a directory containing extraction data.
        Optionally takes a ``mesh`` argument, so that function spaces can be
        established on the same mesh as an existing spline object for
        facilitating segregated solver schemes.  (Splines common to one
        set of extraction data are always treated as a monolothic mixed
        function space.)  This parameter is ignored if ``sourceArg`` is an
        extraction generator, in which case the generator's mesh is always
        used.  Everything to do with the spline is integrated
        using a quadrature rule of degree ``quadDeg``.
        The argument ``doPermutation`` chooses whether or not to apply a
        permutation to the IGA DoF order.  It is ignored if reading
        extraction data from the filesystem.  The argument ``comm`` is an
        MPI communicator that for the object that is ignored if ``sourceArg``
        is a generator.  (In that case, the communicator is the same as that
        of the generator.)
        """

        if isinstance(sourceArg, AbstractExtractionGenerator):
            if mesh != None and mpirank == 0:
                print(
                    "WARNING: Parameter 'mesh' ignored.  Using mesh from "
                    + "extraction generator instead."
                )
            self.initFromGenerator(sourceArg, quadDeg, doPermutation)
        else:
            self.initFromFilesystem(sourceArg, quadDeg, comm, mesh)

        self.genericSetup()

    def initFromGenerator(
        self, generator, quadDeg, doPermutation=DEFAULT_DO_PERMUTATION
    ):
        """
        Generates instance from an ``AbstractExtractionGenerator``, without
        passing through the filesystem.  This mainly exists to circumvent
        broken parallel HDF5 file output for quads and hexes in 2017.2
        (See Issue #1000 for dolfinx on Bitbucket.)

        NOTE: While seemingly-convenient for small-scale testing/demos, and
        more robust in the sense that it makes no assumptions about the
        DoF ordering in FunctionSpaces being deterministic,
        this is not the preferred workflow for most realistic
        cases, as it forces a possibly-expensive preprocessing step to
        execute every time the analysis code is run.
        """

        if doPermutation:
            generator.applyPermutation()

        self.quadDeg = quadDeg
        self.nsd = generator.getNsd()
        self.useDG = generator.useDG()
        self.elementType = generator.extractionElement()
        self.nFields = generator.getNFields()
        self.p_control = generator.getDegree(-1)
        self.p = []
        for i in range(0, self.nFields):
            self.p += [generator.getDegree(i)]
        self.mesh = generator.mesh
        self.cpFuncs = generator.cpFuncs
        self.VE = generator.VE
        self.VE_control = generator.VE_control
        self.V = generator.V
        self.V_control = generator.V_control
        self.M = generator.M
        self.M_control = generator.M_control
        self.comm = generator.getComm()
        zeroDofIS = PETSc.IS(self.comm)
        zeroDofIS.createGeneral(np.array(generator.zeroDofs, dtype=INDEX_TYPE))
        self.zeroDofs = zeroDofIS

    def initFromFilesystem(self, dirname, quadDeg, comm, mesh=None):
        """
        Generates instance from extraction data in directory ``dirname``.
        Optionally takes a ``mesh`` argument, so that function spaces can be
        established on the same mesh as an existing spline object for
        facilitating segregated solver schemes.  (Splines common to one
        set of extraction data are always treated as a monolothic mixed
        function space.)  Everything to do with the spline is integrated
        using a quadrature rule of degree ``quadDeg``.
        """

        self.quadDeg = quadDeg
        self.comm = comm

        # read function space info
        f = open(dirname + "/" + EXTRACTION_INFO_FILE, "r")
        fs = f.read()
        f.close()
        lines = fs.split("\n")
        lineCount = 0
        self.nsd = int(lines[lineCount])
        lineCount += 1
        self.elementType = lines[lineCount]
        lineCount += 1
        self.nFields = int(lines[lineCount])
        lineCount += 1
        self.p_control = int(lines[lineCount])
        lineCount += 1
        ncp_control = int(lines[lineCount])
        lineCount += 1
        self.p = []
        ncp = []
        for i in range(0, self.nFields):
            self.p += [
                int(lines[lineCount]),
            ]
            lineCount += 1
            ncp += [
                int(lines[lineCount]),
            ]
            lineCount += 1

        # prealloc_control = int(lines[lineCount])
        # lineCount += 1
        # prealloc = int(lines[lineCount])

        # # read mesh if none provided
        # # f = HDF5File(mpi_comm_world(),dirname+"/"+EXTRACTION_DATA_FILE,'r')
        # f = dolfinx.HDF5File(self.comm, dirname + "/" +
        #                      EXTRACTION_DATA_FILE, "r")
        if mesh == None:
            with dolfinx.io.XDMFFile(self.comm, dirname + "/" + EXTRACTION_MESH_FILE, "r") as xdmf:
                self.mesh = xdmf.read_mesh()
                print(self.comm.rank, "after")
                # kk = self.mesh.topology.index_map(2)
                print(self.comm.rank, self.mesh.topology.original_cell_index,
                      self.mesh.topology.index_map(2).num_ghosts)
                # print(self.comm.rank, len(
                #     self.mesh.topology.original_cell_index), kk.num_ghosts)
        else:
            self.mesh = mesh

        # create function spaces

        dim = self.mesh.topology.dim
        discontinuous = self.elementType == "DG"

        self.VE_control = createElementType(self.p_control, dim, discontinuous)
        self.V_control = createFunctionSpace(self.mes, self.VE_control)

        self.VE = createVectorElementType(
            self.p, dim, discontinuous, self.nFields)
        self.V = createFunctionSpace(self.mesh, self.VE)

        # read control functions
        viewer = PETSc.ViewerHDF5(self.comm).createBinary(
            dirname + "/" + EXTRACTION_VEC_FILE_CTRL_PTS, "r"
        )
        self.cpFuncs = []
        for i in range(0, self.nsd + 1):
            self.cpFuncs += [
                dolfinx.fem.Function(self.V_control),
            ]
            f.read(self.cpFuncs[i], EXTRACTION_H5_CONTROL_FUNC_NAME(i))
        f.close()

        # read extraction matrix and create transpose for control space
        # FIXME to improve
        Istart, Iend = self.cpFuncs[0].vector.getOwnershipRange()
        nLocalNodes = Iend - Istart
        self.M_control = PETSc.Mat(self.comm)
        self.M_control.create(self.comm)
        # arguments: [[localRows,localColumns],[globalRows,globalColums]]
        # or is it [[localRows,globalRows],[localColumns,globalColums]]?
        # the latter seems to be what comes out of getSizes()...
        if self.comm.size > 1:
            self.M_control.setSizes([[nLocalNodes, None], [None, ncp_control]])

        viewer = PETSc.ViewerHDF5(self.comm).createBinary(
            dirname + "/" + EXTRACTION_MAT_FILE_CTRL, "r"
        )
        self.M_control.load(viewer)
        viewer.destroy()

        # exit()

        # read extraction matrix and create transpose
        self.M = PETSc.Mat(self.comm)
        self.M.create(self.comm)
        # arguments: [[localRows,localColumns],[globalRows,globalColums]]
        if self.comm.size > 1:
            nLocalNodes = self.V.dofmap.index_map.size_local * self.V.dofmap.index_map_bs
            totalDofs = sum(ncp)
            self.M.setSizes([[nLocalNodes, None], [None, totalDofs]])
        # MPETSc2.setType('aij') # sparse
        # MPETSc2.setPreallocationNNZ(prealloc)
        # MPETSc2.setUp()
        viewer = PETSc.ViewerHDF5(self.comm).createBinary(
            dirname + "/" + EXTRACTION_MAT_FILE, "r"
        )
        self.M.load(viewer)
        viewer.destroy()

        # read zero-ed dofs
        # f = open(dirname+"/"+EXTRACTION_ZERO_DOFS_FILE(mpirank),"r")
        # f = open(dirname+"/"+EXTRACTION_ZERO_DOFS_FILE,"r")
        # fs = f.read()
        # f.close()
        # dofStrs = fs.split()
        # zeroDofs  = []
        # for dofStr in dofStrs:
        #    # only keep the ones for this processor
        #    possibleDof = int(dofStr)
        #    if(possibleDof < Iend and possibleDof >= Istart):
        #        zeroDofs += [possibleDof,]
        # self.zeroDofs = PETSc.IS()
        # self.zeroDofs.createGeneral(array(zeroDofs,dtype=INDEX_TYPE))

        viewer = PETSc.ViewerHDF5(self.comm).createBinary(
            dirname + "/" + EXTRACTION_ZERO_DOFS_FILE, "r"
        )
        self.zeroDofs = PETSc.IS(self.comm)
        self.zeroDofs.load(viewer)
        viewer.destroy()

    def genericSetup(self):
        """
        Setup steps to take regardless of the source of extraction data.
        """

        # for marking subdomains
        tag = 0
        fdim = self.mesh.topology.dim - 1
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, fdim)
        all_facets = np.arange(*self.mesh.topology.index_map(fdim).local_range)
        self.boundaryMarkers = dolfinx.mesh.meshtags(
            self.mesh, fdim, all_facets, np.full_like(all_facets, tag))

        # caching transposes of extraction matrices
        if FORM_MT:
            self.MT_control = self.M_control.transpose(PETSc.Mat(self.comm))
            self.MT = self.M.transpose(PETSc.Mat(self.comm))

        # geometrical mapping
        components = []
        for i in range(0, self.nsd):
            components += [
                self.cpFuncs[i] / self.cpFuncs[self.nsd],
            ]
        self.F = ufl.as_vector(components)
        self.DF = ufl.grad(self.F)

        # debug
        # self.DF = Identity(self.nsd)

        # metric tensor
        self.g = getMetric(self.F)  # (self.DF.T)*self.DF

        # normal of pre-image in coordinate chart
        self.N = ufl.FacetNormal(self.mesh)

        # normal that preserves orthogonality w/ pushed-forward tangent vectors
        self.n = mappedNormal(self.N, self.F)

        # integration measures
        self.dx = tIGArxMeasure(volumeJacobian(self.g), ufl.dx, self.quadDeg)
        self.ds = tIGArxMeasure(
            surfaceJacobian(
                self.g, self.N), ufl.ds, self.quadDeg, self.boundaryMarkers
        )

        # useful for defining Cartesian differential operators
        self.pinvDF = pinvD(self.F)

        # useful for tensors given in parametric coordinates
        self.gamma = getChristoffel(self.g)

        self.linsolverOpts = ExtractedSpline.getDefaultPETScSolverOptions()
        self.setSolverOptions()

    def FEtoIGA(self, u):
        """
        This solves the pseudoinverse problem to get the IGA degrees of
        freedom from the finite element ones associated with ``u``, which
        is a ``Function``.  It uses the ``self`` instance's linear solver
        object if available.  The return value is a ``PETScVector`` of the
        IGA degrees of freedom.

        NOTE: This is inefficient and should rarely be necessary.  It is
        mainly intended for testing purposes, or as a last resort.
        """
        tempFunc = dolfinx.fem.Function(self.V)
        tempFunc.assign(u)
        # RHS of problem for initial guess IGA DoFs:
        MTtemp = self.extractVector(tempFunc.vector(), applyBCs=False)
        # Vector with right dimension for IGA DoFs (content doesn't matter):
        tempVec = self.extractVector(tempFunc.vector())
        # LHS of problem for initial guess:
        MTM = self.M.transposeMatMult(self.M)
        if self.linearSolver == None:
            # FIXME
            dolfinx.solve(MTM, tempVec, MTtemp)
        else:
            # FIXME
            self.linearSolver.solve(MTM, tempVec, MTtemp)
        return tempVec

    # def interpolateAsDisplacement(self,functionList=[]):
    #
    #    """
    #    Given a list of scalar functions, get a displacement field from
    #    mesh coordinates to control + functions in physical space,
    #    interpolated on linear elements for plotting without discontinuities
    #    on cut-up meshes. Default argument of ``functionList=[]``
    #    just interpolates the control functions.  If there are fewer elements
    #    in ``functionList`` than there are control functions, then the missing
    #    functions are assumed to be zero.
    #
    #    NOTE: Currently only works with extraction to simplicial elements.
    #    """
    #
    #    #expr = SplineDisplacementExpression(degree=self.quadDeg)
    #    expr = SplineDisplacementExpression\
    #           (element=self.VE_displacement)
    #    expr.spline = self
    #    expr.functionList = functionList
    #    disp = Function(self.V_displacement)
    #    disp.interpolate(expr)
    #    return disp

    # Cartesian differential operators in deformed configuration
    # N.b. that, when applied to tensor-valued f, f is considered to be
    # in the Cartesian coordinates of the physical configuration, NOT in the
    # local coordinate chart w.r.t. which derivatives are taken by FEniCS
    def grad(self, f, F=None):
        """
        Cartesian gradient of ``f`` w.r.t. physical coordinates.
        Optional argument ``F`` can be used to take the gradient assuming
        a different mapping from
        parametric to physical space.  (Default is ``self.F``.)
        """
        if F == None:
            F = self.F
        return cartesianGrad(f, F)

    def div(self, f, F=None):
        """
        Cartesian divergence of ``f`` w.r.t. physical coordinates.
        Optional argument ``F``
        can be used to take the gradient assuming a different mapping from
        parametric to physical space.  (Default is ``self.F``.)
        """
        if F == None:
            F = self.F
        return cartesianDiv(f, F)

    # only applies in 3D, to vector-valued f
    def curl(self, f, F=None):
        """
        Cartesian curl w.r.t. physical coordinates.  Only applies in 3D, to
        vector-valued ``f``.  Optional argument ``F``
        can be used to take the gradient assuming a different mapping from
        parametric to physical space.  (Default is ``self.F``.)
        """
        if F == None:
            F = self.F
        return cartesianCurl(f, F)

    # partial derivatives with respect to curvilinear coordinates; this is
    # just a wrapper for FEniCS grad(), but included to allow for writing
    # clear, unambiguous scripts
    def parametricGrad(self, f):
        """
        Gradient of ``f`` w.r.t. parametric coordinates.  (Equivalent to UFL
        ``grad()``, but introduced to avoid confusion with ``self.grad()``.)
        """
        return ufl.grad(f)

    # curvilinear variants; if f is only a regular tensor, will create a
    # CurvilinearTensor w/ all indices lowered.  Metric defaults to one
    # generated by mapping self.F (into Cartesian space) if no metric is
    # supplied via f.
    def GRAD(self, f):
        """
        Covariant derivative of a ``CurvilinearTensor``, ``f``, taken w.r.t.
        parametric coordinates, assuming that components
        of ``f`` are also given in this coordinate system.  If a regular tensor
        is passed for ``f``, a ``CurvilinearTensor`` will be created with all
        lowered indices.
        """
        if not isinstance(f, CurvilinearTensor):
            ff = CurvilinearTensor(f, self.g)
        else:
            ff = f
        return curvilinearGrad(ff)

    def DIV(self, f):
        """
        Curvilinear divergence operator corresponding to ``self.GRAD()``.
        Contracts new lowered index from ``GRAD`` with last raised
        index of ``f``.
        If a regular tensor is passed for ``f``, a ``CurvilinearTensor``
        will be created with all raised indices.
        """
        if not isinstance(f, CurvilinearTensor):
            ff = CurvilinearTensor(f, self.g).sharp()
        else:
            ff = f
        return curvilinearDiv(ff)

    # def spatialExpression(self,expr):
    #    """
    #    Converts string ``expr`` into an ``Expression``,
    #    treating the coordinates ``'x[i]'`` in ``expr`` as
    #    spatial coordinates.
    #    (Using the standard ``Expression`` constructor, these would be treated
    #    as parametric coordinates.)
    #
    #    NOTE: Only works when extracting to simplicial elements.
    #    """
    #    retval = tIGArxExpression(degree=self.quadDeg)
    #    retval.expr = Expression(expr,degree=self.quadDeg)
    #    retval.nsd = self.nsd
    #    retval.cpFuncs = self.cpFuncs
    #    return retval

    def parametricExpression(self, expr):
        """
        Create an ``Expression`` from a string, ``expr``, interpreting the
        coordinates ``'x[i]'`` in ``expr`` as parametric coordinates.
        Uses quadrature degree of spline object for interpolation degree.
        """
        return dolfinx.Expression(expr, degree=self.quadDeg)

    def parametricCoordinates(self):
        """
        Wrapper for ``SpatialCoordiantes()`` to avoid confusion, since
        FEniCS's spatial coordinates are used in tIGArx as parametric
        coordinates.
        """
        return ufl.SpatialCoordinate(self.mesh)

    def spatialCoordinates(self):
        """
        Returns the mapping ``self.F``, which gives the spatial coordinates
        of a parametric point.
        """
        return self.F

    def rationalize(self, u):
        """
        Divides its argument ``u`` by the weighting function of the spline's
        control mesh.
        """
        return u / (self.cpFuncs[self.nsd])

    # split out to implement contact
    def extractVector(self, b, applyBCs=True):
        """
        Apply extraction to an FE vector ``b``.  The Boolean ``applyBCs``
        indicates whether or not to apply BCs to the vector.
        """
        # MT determines parallel partitioning of MTb
        if FORM_MT:
            MTb = (self.MT) * b
        else:
            MTb = multTranspose(self.M, b)

        # apply zero bcs to MTAM and MTb
        if applyBCs:
            MTb.setValues(
                self.zeroDofs, np.zeros(self.zeroDofs.getLocalSize())
            )
        MTb.assemble()

        return MTb

    def assembleVector(self, form, applyBCs=True):
        """
        Assemble M^T*b, where ``form`` is a linear form and the Boolean
        ``applyBCs`` indicates whether or not to apply the Dirichlet BCs.
        """
        # b = PETScVector()
        # assemble(form, tensor=b)
        b = petsc_assemble_vector(dolfinx.fem.form(form))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                      mode=PETSc.ScatterMode.REVERSE)

        MTb = self.extractVector(b, applyBCs=applyBCs)

        return MTb

    # split out to implement contact
    def extractMatrix(self, A, applyBCs=True, diag=1):
        """
        Apply extraction to an FE matrix ``A``.  The Boolean ``applyBCs``
        indicates whether or not to apply BCs to the matrix, and the
        optional argument ``diag`` is what will be filled into diagonal
        entries where Dirichlet BCs are applied.
        """
        if FORM_MT:
            MTA = self.MT.matMult(A)
            MTAM = MTA.matMult(self.M)
        else:
            # Needs recent version of petsc4py; seems to be standard version
            # used in docker container, though, since this function works
            # fine on stampede.
            MTAM = A.PtAP(self.M)

        # apply zero bcs to MTAM and MTb
        # (default behavior is to set diag=1, as desired)
        if applyBCs:
            MTAM.zeroRowsColumns(self.zeroDofs, diag)

        MTAM.assemble()

        return MTAM

    def assembleMatrix(self, form, applyBCs=True, diag=1):
        """
        Assemble M^T*A*M, where ``form`` is a bilinear form and the Boolean
        ``applyBCs`` indicates whether or not to apply the Dirichlet BCs.
        The optional argument ``diag`` is what will be filled into diagonal
        entries where Dirichlet BCs are applied.  For eigenvalue problems,
        it can be useful to have non-default ``diag``, to move eigenvalues
        corresponding to the Dirichlet BCs.
        """
        A = petsc_assemble_matrix(dolfinx.fem.form(form))
        A.assemble()

        MTAM = self.extractMatrix(A, applyBCs=applyBCs, diag=diag)

        return MTAM

    def assembleLinearSystem(self, lhsForm, rhsForm, applyBCs=True):
        """
        Assembles a linear system corresponding the LHS form ``lhsForm`` and
        RHS form ``rhsForm``.  The optional argument ``applyBCs`` is a
        Boolean indicating whether or not to apply the spline's
        homogeneous Dirichlet BCs.
        """

        # really an unnecessary function now that i split out the
        # vector and matrix assembly...
        return (
            self.assembleMatrix(lhsForm, applyBCs),
            self.assembleVector(rhsForm, applyBCs),
        )

    @staticmethod
    def getDefaultPETScSolverOptions():
        """Gets the default options for PETSc KSP solver.
        Defaults are: conjugate gradient solver with Jacobi preconditioner,
        relative and absolute tolerances are set to ``DEFAULT_LINSOLVER_REL_TOL``
        and ``DEFAULT_LINSOLVER_ABS_TOLS``, respectively, and maximum number of iterations set to ``DEFAULT_LINSOLVER_MAX_ITERS``.

        Returns:
            (str, optional): Default options.
        """

        opts = {}
        opts["ksp_type"] = "cg"  # "gmres", "preonly"
        opts["pc_type"] = "jacobi"  # "none", "lu"
        opts["ksp_rtol"] = DEFAULT_LINSOLVER_REL_TOL
        opts["ksp_atol"] = DEFAULT_LINSOLVER_ABS_TOL
        opts["ksp_max_it"] = DEFAULT_LINSOLVER_MAX_ITERS

        return opts

    def createPETScSolverOptions(self, problem_prefix, linsolverOpts=None):
        """Creates PETSc KSP solver options and associates them
        to the given prefix.

        Args:
            problem_prefix (str): Prefix string to be associated to the options.
            linsolverOpts (dict): List of solver options.

        Returns:
            petsc4py.PETSc.Options: PETSc options for the solver.
        """

        if linsolverOpts is None:
            linsolverOpts = self.linsolverOpts

        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        for k, v in linsolverOpts.items():
            opts[k] = v
        opts.prefixPop()

        return opts

    def createPETScOptionsPrefix(self):
        """Creates the options prefix string for PETSc solver.
        """
        return f"dolfinx_solve_{id(self)}"

    def createPETScSolver(self, comm, linsolverOpts):
        """
        Creates a PETSc solver based on the ``linsolverOpts``.
        """

        solver = PETSc.KSP().create(comm)
        # solver.setOperators(MTAM)

        # Give PETSc solver options a unique prefix
        problem_prefix = self.createPETScOptionsPrefix()
        solver.setOptionsPrefix(problem_prefix)

        # Set PETSc options
        self.createPETScSolverOptions(problem_prefix, linsolverOpts)

        solver.setFromOptions()

        return solver

    def solveLinearSystem(self, MTAM, MTb, u, linsolverOpts=None):
        """
        Solves a linear system of the form

        ``MTAM*U = MTb``

        where ``MTAM`` is the IGA LHS, ``U`` is the vector of IGA unknowns
        (in the homogeneous coordinate representation, if rational splines
        are being used), and ``MTb`` is the IGA RHS.  The FE representation
        of the solution is then the ``Function`` ``u`` which has a vector
        of coefficients given by ``M*U``.  The return value of the function
        is ``U``, as a dolfinx ``PETScVector``.
        PETSc linear solver may be set at ``solver_opts``.
        """

        U = u.vector
        if FORM_MT:
            MTU = self.MT * U
        else:
            MTU = multTranspose(self.M, U)
        MTU.assemble()

        solver = self.createPETScSolver(
            u.function_space.mesh.comm, linsolverOpts)
        solver.setOperators(MTAM)

        # Give PETSc solver options a unique prefix
        problem_prefix = self.createPETScOptionsPrefix()

        # Set matrix and vector PETSc options
        MTAM.setOptionsPrefix(problem_prefix)
        MTAM.setFromOptions()
        MTb.setOptionsPrefix(problem_prefix)
        MTb.setFromOptions()

        # Apply boundary conditions to the rhs
        # apply_lifting(self._b, [self._a], bcs=[self.bcs])
        # self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        # set_bc(self._b, self.bcs)

        # Solve linear system and update ghost values in the solution
        solver.solve(MTb, MTU)
        MTU.assemble()
        # self.u.x.scatter_forward()

        # if self.linearSolver == None:
        #     dolfinx.solve(MTAM, MTU, MTb)
        # else:
        #     self.linearSolver.solve(MTAM, MTU, MTb)

        MMTU = self.M * MTU
        size_local = u.x.index_map.size_local * u.x.block_size
        u.x.array[:size_local] = MMTU.array_r
        u.x.scatter_forward()

        return MTU

    def solveLinearVariationalProblem(self, residualForm, u, applyBCs=True, linsolverOpts=None):
        """
        Solves a linear variational problem with residual ``residualForm'',
        putting the solution in the ``Function`` ``u``.  Homogeneous
        Dirichlet BCs from ``self`` can be optionally applied, based on the
        Boolean parameter ``applyBCs``.  Note that ``residualForm`` may also
        be given in the form of a UFL ``Equation``, e.g., ``lhs==rhs``.  The
        return value of the function is the vector of IGA degrees of freedom,
        as a dolfinx ``PETScVector``.
        PETSc solver options may be set at ``linsolverOpts``.
        """

        if isinstance(residualForm, ufl.equation.Equation):
            lhsForm = residualForm.lhs
            rhsForm = residualForm.rhs
        else:
            # TODO: Why is this so much slower?
            lhsForm = ufl.lhs(residualForm)
            rhsForm = ufl.rhs(residualForm)

        if rhsForm.integrals() == ():
            v = ufl.TestFunction(self.V)
            rhsForm = dolfinx.Constant(0.0) * v[0] * self.dx

        MTAM, MTb = self.assembleLinearSystem(lhsForm, rhsForm, applyBCs)

        x = self.solveLinearSystem(MTAM, MTb, u, linsolverOpts)

        # self._solver.destroy()
        # MTAM.destroy()
        # MTb.destroy()
        # x.destroy()

        return x

    def setSolverOptions(self, maxIters=20, relativeTolerance=1e-5):
        """
        Sets some solver options for the ``ExtractedSpline`` instance, to be
        used in ``self.solve*VariationalProblem()``.
        """
        self.maxIters = maxIters
        self.relativeTolerance = relativeTolerance

    def solveNonlinearVariationalProblem(
        self, residualForm, J, u, referenceError=None, igaDoFs=None, linsolverOpts=None
    ):
        """
        Solves a nonlinear variational problem with residual given by
        ``residualForm``.  ``J`` is the functional derivative of
        the residual w.r.t. the solution, ``u``, or some user-defined
        approximation thereof.  Optionally, a given ``referenceError`` can be
        used instead of the initial residual norm to compute relative errors.
        Optionally, an initial guess can be provided directly as a dolfinx
        ``PETScVector`` of IGA degrees of freedom, which will override
        whatever is in the vector of FE coefficients for ``u`` as the
        initial guess.  If this argument is passed, it is also overwritten
        by the IGA degrees of freedom for the nonlinear problem's solution.
        PETSc solver options may be set at ``linsolverOpts``.
        """
        returningDoFs = not isinstance(igaDoFs, type(None))
        if returningDoFs:
            # Overwrite content of u with extraction of igaDoFs.
            aux = self.M * igaDoFs
            u.x.array[:] = aux.array_r
            u.x.scatter_forward()

        # Newton iteration loop:
        converged = False
        for i in range(0, self.maxIters):
            MTAM, MTb = self.assembleLinearSystem(J, residualForm)
            currentNorm = MTb.norm(PETSc.NormType.NORM_2)
            if i == 0 and referenceError == None:
                referenceError = currentNorm
            relativeNorm = currentNorm / referenceError
            if self.comm.Get_rank() == 0:
                print(
                    "Solver iteration: "
                    + str(i)
                    + " , Relative norm: "
                    + str(relativeNorm)
                )
                sys.stdout.flush()
            if relativeNorm < self.relativeTolerance:
                converged = True
                break
            du = dolfinx.fem.Function(self.V)
            igaIncrement = self.solveLinearSystem(MTAM, MTb, du, linsolverOpts)
            u.x.array[:] = u.x.array - du.x.array
            u.x.scatter_forward()
            if returningDoFs:
                igaDoFs -= igaIncrement
        if not converged:
            print("ERROR: Nonlinear solver failed to converge.")
            exit()

    # project a scalar onto a given space
    def projectScalarOnto(self, toProject, V, linsolverOpts=None, lumpMass=False):
        """
        L2 projection of some UFL object ``toProject`` onto a given space ``V``
        scalar FE functions.  The optional
        parameter ``lumpMass`` is a Boolean indicating whether or not to
        lump mass in  the projection.  The optional parameter ``linearSolver``
        indicates what linear solver to use, choosing the default if
        ``None`` is passed.
        PETSc solver options may be set at ``linsolverOpts``.
        """
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        # don't bother w/ change of variables in integral
        # res = inner(u-toProject,v)*self.dx.meas

        # Note: for unclear reasons, extracting the lhs/rhs from the
        # residual is both itself very slow, and also causes the assembly
        # to become very slow.

        if lumpMass:
            lhsForm = ufl.inner(1.0, v) * self.dx.meas
            A = petsc_assemble_vector(dolfinx.fem.form(lhsForm))
        else:
            lhsForm = ufl.inner(u, v) * self.dx.meas  # lhs(res)
            A = petsc_assemble_matrix(dolfinx.fem.form(lhsForm))

        rhsForm = ufl.inner(toProject, v) * self.dx.meas  # rhs(res)
        b = petsc_assemble_vector(dolfinx.fem.form(rhsForm))

        A.assemble()
        b.assemble()

        u = dolfinx.fem.Function(V)

        # solve(A,u.vector(),b)
        if lumpMass:
            u.vector.pointwiseDivide(b.vector, A.vector)
        else:
            comm = V.mesh.comm
            solver = self.createPETScSolver(comm, linsolverOpts)
            solver.setOperators(A)

            # Give PETSc solver options a unique prefix
            problem_prefix = self.createPETScOptionsPrefix()

            # Set matrix and vector PETSc options
            A.setOptionsPrefix(problem_prefix)
            A.setFromOptions()
            b.setOptionsPrefix(problem_prefix)
            b.setFromOptions()

            # Solve linear system and update ghost values in the solution
            solver.solve(b, u.vector)
            u.x.scatter_forward()

        return u

    # project a scalar onto linears for plotting
    def projectScalarOntoControl(self, toProject, linsolverOpts=None, lumpMass=False):
        """
        L2 projection of some UFL object ``toProject`` onto a space of the control mesh,
        scalar FE functions (typically used for plotting).  The optional
        parameter ``lumpMass`` is a Boolean indicating whether or not to
        lump mass in  the projection.  The optional parameter ``linearSolver``
        indicates what linear solver to use, choosing the default if
        ``None`` is passed.
        PETSc solver options may be set at ``linsolverOpts``.
        """
        return self.projectScalarOnto(toProject, self.V_control, linsolverOpts, lumpMass)

    # project something onto the solution space; ignore bcs by default
    def project(self, toProject, applyBCs=False, rationalize=True, lumpMass=False):
        """
        L2 projection of some UFL object ``toProject`` onto the
        ``ExtractedSpline`` object's solution space.  Can optionally apply
        homogeneous Dirichlet BCs with the Boolean parameter
        ``applyBCs``.  By default, no BCs are applied in projection.  The
        parameter ``rationalize`` is a Boolean specifying whether or not
        to divide through by the weight function (thus returning a UFL
        ``Division`` object, rather than a ``Function``).  Rationalization
        is done by default.  The parameter ``lumpMass`` is a Boolean
        indicating whether or not to use mass lumping, which is ``False`` by
        default.  (Lumping may be useful for preventing oscillations when
        projecting discontinuous initial/boundary conditions.)
        """

        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        u = self.rationalize(u)
        v = self.rationalize(v)
        rhsForm = ufl.inner(toProject, v) * self.dx
        retval = dolfinx.fem.Function(self.V)
        if not lumpMass:
            lhsForm = ufl.inner(u, v) * self.dx
            self.solveLinearVariationalProblem(
                lhsForm == rhsForm, retval, applyBCs)
        else:
            if self.nFields == 1:
                oneConstant = dolfinx.Constant(1.0)
            else:
                oneConstant = dolfinx.Constant(self.nFields * (1.0,))

            assert False, "To be revisited for dolfinx"
            lhsForm = ufl.inner(oneConstant, v) * self.dx
            lhsVecFE = dolfinx.assemble(lhsForm)
            lhsVec = self.extractVector(lhsVecFE, applyBCs=False)
            rhsVecFE = dolfinx.assemble(rhsForm)
            rhsVec = self.extractVector(rhsVecFE, applyBCs=applyBCs)
            igaDoFs = self.extractVector(dolfinx.fem.Function(self.V).vector())
            igaDoFs.pointwiseDivide(
                rhsVec.vector,
                lhsVec.vector
            )
            retval.vector()[:] = self.M * igaDoFs
        if rationalize:
            retval = self.rationalize(retval)
        return retval


class AbstractCoordinateChartSpline(AbstractExtractionGenerator):
    """
    This abstraction represents a spline whose parametric
    coordinate system consists of a
    using a single coordinate chart, so coordinates provide a unique set
    of basis functions; this applies to single-patch B-splines, T-splines,
    NURBS, etc., and, with a little creativity, can be stretched to cover
    multi-patch constructions.
    """

    @ abc.abstractmethod
    def getNodesAndEvals(self, x, field):
        """
        Given a parametric point ``x``, return a list of the form

        ``[[index0, N_index0(x)], [index1,N_index1(x)], ... ]``

        where ``N_i`` is the ``i``-th basis function of the scalar polynomial
        spline space (NOT of the rational space) corresponding to a given
        ``field``.
        """
        return

    # return a matrix M for extraction
    def generateM_control(self):
        """
        Generates the extraction matrix for the single scalar spline space
        used to represent all homogeneous components of the mapping ``F``
        from parametric to physical space.
        """
        return self.impl_generateM(True)

    def generateM(self):
        """
        Generates the extraction matrix for the mixed function space of
        all unknown scalar fields.
        """
        return self.impl_generateM(False)

    def impl_generateM(self, is_control):
        """
        Generates the extraction matrix for either the scalar spline space
        used to represent all homogeneous components of the mapping ``F``
        (is_control=True) of for the mixed function space of all unknown
        scalar fields (is_control=False).
        """

        MPETSc = PETSc.Mat(self.comm)

        # MPETSc.create(PETSc.COMM_WORLD)
        # arguments: [[localRows,localColumns],[globalRows,globalColums]]
        # MPETSc.setSizes([[nLocalNodes,None],[None,self.getNcp(-1)]])
        # MPETSc.setType('aij') # sparse

        # MPETSc.create()

        nFields = 1 if is_control else self.getNFields()
        dim = self.mesh.geometry.dim

        V = self.V_control if is_control else self.V
        is_scalar = V.num_sub_spaces == 0

        bs = V.dofmap.index_map_bs
        nLocalFEMNodes = V.dofmap.index_map.size_local * bs

        if is_control:
            nGlobalIGADofs = self.getNcp(-1)
        else:
            nGlobalIGADofs = sum([self.getNcp(i) for i in range(nFields)])

        MPETSc.createAIJ(
            [[nLocalFEMNodes, None], [None, nGlobalIGADofs]], comm=self.comm)
        MPETSc.setPreallocationNNZ(
            [self.getPrealloc(is_control), self.getPrealloc(is_control)])

        # just slow down quietly if preallocation is insufficient
        MPETSc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        # for debug:
        # MPETSc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, True)
        MPETSc.setUp()

        for field in range(0, nFields):

            if is_scalar:
                Vf = V
                local_range = Vf.dofmap.index_map.local_range
                fem_dofs_range = range(*local_range)
            else:

                Vf, fem_dofs_range = V.sub(field).collapse()
                size_local = Vf.dofmap.index_map.size_local
                fem_dofs_range = np.array(
                    fem_dofs_range[0:size_local]) + V.dofmap.index_map.local_range[0] * bs

            igaDofsOffset = sum([self.getNcp(i) for i in range(field)])

            x_nodes = Vf.tabulate_dof_coordinates()

            for i, fem_dof in enumerate(fem_dofs_range):
                x = x_nodes[i, :dim]
                nodesAndEvals = self.getNodesAndEvals(
                    x, -1 if is_control else field)

                # cols = array(nodesAndEvals,dtype=INDEX_TYPE)[:,0]
                # rows = array([matRow,],dtype=INDEX_TYPE)
                # values = npTranspose(array(nodesAndEvals)[:,1:2])
                # MPETSc.setValues(rows,cols,values,addv=PETSc.InsertMode.INSERT)

                ignore_eps = self.getIgnoreEps()
                # FIXME: to vectorize
                for dof_col_local, val in nodesAndEvals:
                    if abs(val) > ignore_eps:
                        MPETSc[fem_dof, dof_col_local + igaDofsOffset] = val

        MPETSc.assemble()

        return MPETSc

    # override default behavior to order unknowns according to what task's
    # FE mesh they overlap.  this will (hopefully) reduce communication
    # cost in the matrix--matrix multiplies

    def generatePermutation(self):
        """
        Generates a permutation of the IGA degrees of freedom that tries to
        ensure overlap of their parallel partitioning with that of the FE
        degrees of freedom, which are partitioned automatically based on the
        FE mesh.
        """

        # MPETSc.create(PETSc.COMM_WORLD)
        # arguments: [[localRows,localColumns],[globalRows,globalColums]]
        # MPETSc.setSizes([[nLocalNodes,None],[None,self.getNcp(-1)]])
        # MPETSc.setType('aij') # sparse

        # MPETSc.create()

        mpirank = self.comm.rank

        nFields = self.getNFields()
        dim = self.mesh.geometry.dim

        is_scalar = self.V.num_sub_spaces == 0

        bs = self.V.dofmap.index_map_bs
        nLocalFEMNodes = self.V.dofmap.index_map.size_local * bs

        nGlobalIGADofs = sum([self.getNcp(i) for i in range(nFields)])

        MPETSc = PETSc.Mat(self.comm)
        MPETSc.createAIJ(
            [[nLocalFEMNodes, None], [None, nGlobalIGADofs]], comm=self.comm)
        MPETSc.setPreallocationNNZ(
            [self.getPrealloc(False), self.getPrealloc(False)])
        MPETSc.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
        MPETSc.setUp()

        for field in range(0, nFields):

            if is_scalar:
                Vf = self.V
                local_range = Vf.dofmap.index_map.local_range
                fem_dofs_range = range(*local_range)
            else:

                Vf, fem_dofs_range = self.V.sub(field).collapse()
                size_local = Vf.dofmap.index_map.size_local
                fem_dofs_range = np.array(
                    fem_dofs_range[0:size_local]) + self.V.dofmap.index_map.local_range[0] * bs

            igaDofsOffset = sum([self.getNcp(i) for i in range(field)])

            x_nodes = Vf.tabulate_dof_coordinates()

            for i, fem_dof in enumerate(fem_dofs_range):
                x = x_nodes[i, :dim]
                nodesAndEvals = self.getNodesAndEvals(x, field)

                # cols = array(nodesAndEvals,dtype=INDEX_TYPE)[:,0]
                # rows = array([matRow,],dtype=INDEX_TYPE)
                # values = npTranspose(array(nodesAndEvals)[:,1:2])
                # MPETSc.setValues(rows,cols,values,addv=PETSc.InsertMode.INSERT)

                for dof_col_local, _ in nodesAndEvals:
                    MPETSc[fem_dof, dof_col_local +
                           igaDofsOffset] = (mpirank + 1)  # need to avoid losing zeros...

        MPETSc.assemble()

        MT = MPETSc.transpose(PETSc.Mat(self.comm))
        Istart, Iend = MT.getOwnershipRange()
        nLocal = Iend - Istart
        partitionInts = np.zeros(nLocal, dtype=INDEX_TYPE)
        for i in np.arange(Istart, Iend):
            rowValues = MT.getRow(i)[1]
            # isolate nonzero entries (likely not needed ?)
            rowValues = np.extract(rowValues > 0, rowValues)
            if rowValues.size == 0:
                continue
            iLocal = i - Istart
            modeValues = sp.stats.mode(rowValues)[0]
            partitionInts[iLocal] = int(modeValues - 0.5)
        partitionIS = PETSc.IS(self.comm)
        partitionIS.createGeneral(partitionInts, comm=self.comm)

        # kludgy, non-scalable solution:

        # all-gather the partition indices and apply argsort to their
        # underlying arrays
        bigIndices = np.argsort(
            partitionIS.allGather().getIndices()).astype(INDEX_TYPE)

        # note: index set sort method only sorts locally on each processor

        # note: output of argsort is what we want for MatPermute(); it
        # maps from indices in the sorted array, to indices in the original
        # unsorted array.

        # use slices [Istart:Iend] of the result from argsort to create
        # a new IS that can be used as a petsc ordering
        retval = PETSc.IS(self.comm)
        retval.createGeneral(bigIndices[Istart:Iend], comm=self.comm)

        return retval


# abstract class representing a scalar basis of functions on a manifold for
# which we assume that each point has unique coordinates.
class AbstractScalarBasis(object):
    """
    Abstraction defining the behavior of a collection of scalar basis
    functions, defined on a manifold for which each point has unique
    coordinates.
    """

    __metaclass__ = abc.ABCMeta

    @ abc.abstractmethod
    def getNodesAndEvals(self, xi):
        """
        Given a parametric point ``xi``, return a list of the form

        ``[[index0, N_index0(xi)], [index1,N_index1(xi)], ... ]``

        where ``N_i`` is the ``i``-th basis function.
        """
        return

    @ abc.abstractmethod
    def getNcp(self):
        """
        Returns the total number of basis functions.
        """
        return

    @ abc.abstractmethod
    def generateMesh(self, comm=worldcomm):
        """
        Generates and returns an FE mesh sufficient for extracting this spline
        basis.  The argument ``comm`` indicates what MPI communicator to
        partition the mesh across, defaulting to ``MPI_COMM_WORLD``.
        """
        return

    @ abc.abstractmethod
    def getDegree(self):
        """
        Returns a polynomial degree for FEs that is sufficient for extracting
        this spline basis.
        """
        return

    # TODO: get rid of the DG stuff in coordinate chart splines, since
    # getNodesAndEvals() is inherently unstable for discontinuous functions
    # and some other instantiation of AbstractExtractionGenerator
    # is needed to reliably handle $C^{-1}$ splines.

    # @abc.abstractmethod
    # assume DG unless this is overridden by a subclass (as DG will work even
    # if CG is okay (once they fix DG for quads/hexes at least...))
    def needsDG(self):
        """
        Returns a Boolean indicating whether or not DG elements are needed
        to represent this spline space (i.e., whether or not the basis is
        discontinuous).
        """
        return True

    # @abc.abstractmethod
    # def getParametricDimension(self):
    #    return

    # Override this in subclasses to optimize memory use.  It should return
    # the maximum number of IGA basis functions whose supports might contain
    # a finite element node (i.e, the maximum number of nonzero
    # entries in a row of M corrsponding to that FE basis function.)
    def getPrealloc(self):
        """
        Returns some upper bound on the number of nonzero entries per row
        of the extraction matrix for this spline space.  If this can be
        easily estimated for a specific spline type, then this method
        should almost certainly be overriden by that subclass for memory
        efficiency, as the default value implemented in the abstract class is
        overkill.
        """
        return DEFAULT_PREALLOC


# interface needed for a control mesh with a coordinate chart
class AbstractControlMesh(object):
    """
    Abstraction representing the behavior of a control mesh, i.e., a mapping
    from parametric to physical space.
    """

    __metaclass__ = abc.ABCMeta

    @ abc.abstractmethod
    def getHomogeneousCoordinate(self, node, direction):
        """
        Returns the ``direction``-th homogeneous component of the control
        point with index ``node``.
        """
        return

    @ abc.abstractmethod
    def getScalarSpline(self):
        """
        Returns the instance of ``AbstractScalarBasis`` that represents
        each homogeneous component of the control mapping.
        """
        return

    @ abc.abstractmethod
    def getNsd(self):
        """
        Returns the dimension of physical space.
        """
        return


class AbstractMultiFieldSpline(AbstractCoordinateChartSpline):
    """
    Interface for a general multi-field spline.  The reason this is
    a special case of ``AbstractCoordinateChartSpline``
    (instead of being redundant in light of AbstractExtractionGenerator)
    is that it uses a collection of ``AbstractScalarBasis`` objects, whose
    ``getNodesAndEvals()`` methods require parametric coordinates
    to correspond to unique points.
    """

    __metaclass__ = abc.ABCMeta

    @ abc.abstractmethod
    def getControlMesh(self):
        """
        Returns some object implementing ``AbstractControlMesh``, that
        represents this spline's control mesh.
        """
        return

    @ abc.abstractmethod
    def getFieldSpline(self, field):
        """
        Returns the ``field``-th unknown scalar field's
        ``AbstractScalarBasis``.
        """
        return

    # overrides method inherited from AbstractExtractionGenerator, using
    # getPrealloc() methods from its AbstractScalarBasis members.
    def getPrealloc(self, control):
        if control:
            retval = self.getScalarSpline(-1).getPrealloc()
        else:
            maxPrealloc = 0
            for i in range(0, self.getNFields()):
                prealloc = self.getScalarSpline(i).getPrealloc()
                if prealloc > maxPrealloc:
                    maxPrealloc = prealloc
            retval = maxPrealloc
        # print control, retval
        return retval

    def getScalarSpline(self, field):
        """
        Returns the ``field``-th unknown scalar field's \
        ``AbstractScalarBasis``, or, if ``field==-1``, the
        basis for the scalar space of the control mesh.
        """
        if field == -1:
            return self.getControlMesh().getScalarSpline()
        else:
            return self.getFieldSpline(field)

    def getNsd(self):
        """
        Returns the dimension of physical space.
        """
        return self.getControlMesh().getNsd()

    def getHomogeneousCoordinate(self, node, direction):
        """
        Invokes the synonymous method of its control mesh.
        """
        return self.getControlMesh().getHomogeneousCoordinate(node, direction)

    def getNodesAndEvals(self, x, field):
        return self.getScalarSpline(field).getNodesAndEvals(x)

    def generateMesh(self):
        return self.getScalarSpline(-1).generateMesh(comm=self.comm)

    def getDegree(self, field):
        """
        Returns the polynomial degree needed to extract the ``field``-th
        unknown scalar field.
        """
        return self.getScalarSpline(field).getDegree()

    def getNcp(self, field):
        """
        Returns the number of degrees of freedom for a given ``field``.
        """
        return self.getScalarSpline(field).getNcp()

    def useDG(self):
        for i in range(-1, self.getNFields()):
            if self.getScalarSpline(i).needsDG():
                return True
        return False


# common case of all control functions and fields belonging to the
# same scalar space.  Note: fields are all stored in homogeneous format, i.e.,
# they need to be divided through by weight to get an iso-parametric
# formulation.
class EqualOrderSpline(AbstractMultiFieldSpline):
    """
    A concrete subclass of ``AbstractMultiFieldSpline`` to cover the common
    case of multi-field splines in which all unknown scalar fields are
    discretized using the same ``AbstractScalarBasis``.
    """

    # args: numFields, controlMesh
    def customSetup(self, args):
        """
        ``args = (numFields,controlMesh)``, where ``numFields`` is the
        number of unknown scalar fields and ``controlMesh`` is an
        ``AbstractControlMesh`` providing the mapping from parametric to
        physical space and, in this case, the scalar basis to be used for
        all unknown scalar fields.
        """
        self.numFields = args[0]
        self.controlMesh = args[1]

    def getNFields(self):
        return self.numFields

    def getControlMesh(self):
        return self.controlMesh

    def getFieldSpline(self, field):
        return self.getScalarSpline(-1)

    def addZeroDofsByLocation(self, subdomain, field):
        """
        Because, in the equal-order case, there is a one-to-one
        correspondence between the DoFs of the scalar fields and the
        control points of the geometrical mapping, one may, in some cases,
        want to assign boundary conditions to the DoFs of the scalar fields
        based on the locations of their corresponding control points.

        This method assigns homogeneous Dirichlet BCs to DoFs of a given
        ``field`` if the corresponding control points fall within
        ``subdomain``, which is an instance of ``SubDomain``.
        """

        # this is prior to the permutation
        Istart, Iend = self.M_control.getOwnershipRangeColumn()
        nsd = self.getNsd()
        # since this checks every single control point, it needs to
        # be scalable
        p = np.zeros(nsd + 1)
        for I in np.arange(Istart, Iend):
            for j in np.arange(0, nsd + 1):
                p[j] = self.getHomogeneousCoordinate(I, j)
            for j in np.arange(0, nsd):
                p[j] /= p[nsd]
            # make it strictly based on location, regardless of how the
            # on_boundary argument is handled
            isInside = subdomain(p[0:nsd], False) or subdomain(p[0:nsd], True)
            if isInside:
                self.zeroDofs += [
                    self.globalDof(field, I),
                ]


# a concrete case with a list of distinct scalar splines
class FieldListSpline(AbstractMultiFieldSpline):
    """
    A concrete case of a multi-field spline that is constructed from a given
    list of ``AbstractScalarBasis`` objects.
    """

    # args: controlMesh, fields
    def customSetup(self, args):
        """
        ``args = (controlMesh,fields)``, where ``controlMesh`` is an
        ``AbstractControlMesh`` providing the mapping from parametric to
        physical space and ``fields`` is a list of ``AbstractScalarBasis``
        objects for the unknown scalar fields.
        """
        self.controlMesh = args[0]
        self.fields = args[1]

    def getNFields(self):
        return len(self.fields)

    def getControlMesh(self):
        return self.controlMesh

    def getFieldSpline(self, field):
        return self.fields[field]
