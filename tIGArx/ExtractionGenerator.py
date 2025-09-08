import abc
import time

import numpy as np

import dolfinx
from petsc4py import PETSc

from tigarx.common import (
    selfcomm,
    worldcomm,
    USE_DG_DEFAULT,
    DEFAULT_PREALLOC,
    DEFAULT_BASIS_FUNC_IGNORE_EPS,
    FORM_MT,
    INDEX_TYPE,
    DEFAULT_DO_PERMUTATION
)
from tigarx.utils import (
    createElementType,
    createVectorElementType,
    generateIdentityPermutation,
    multTranspose,
)


class AbstractExtractionGenerator(object):
    """
    Abstract class representing the minimal set of functions needed to write
    extraction operators for a spline.
    """

    __metaclass__ = abc.ABCMeta

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

    def __init__(self, comm, *args):
        """
        Arguments in ``*args`` are passed as a tuple to
        ``self.customSetup()``.  Appropriate arguments vary by subclass.  If
        the first argument ``comm`` is of type ``petsc4py.PETSc.Comm``, then
        it will be treated as a communicator for the extraction generator.
        Otherwise, it is treated as if it were the first argument in ``args``.
        """

        if type(comm) is not type(selfcomm):
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

    def genericSetup(self, profile=False):
        """
        Common setup steps for all subclasses (called in ``self.__init__()``).
        """

        self.mesh = self.generateMesh()

        # note: if self.nsd is set in a customSetup, then the subclass
        # getNsd() references that, this is still safe
        self.nsd = self.getNsd()

        self.VE_control = self.createElementType(self.getDegree(-1))
        self.V_control = dolfinx.fem.functionspace(self.mesh, self.VE_control)

        nFields = self.getNFields()
        degrees = [self.getDegree(i) for i in range(nFields)]
        self.VE = self.createVectorElementType(degrees, nFields)
        self.V = dolfinx.fem.functionspace(self.mesh, self.VE)

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
                MTC = MT_control * self.cpFuncs[i].x.petsc_vec
            else:
                MTC = multTranspose(self.M_control, self.cpFuncs[i].x.petsc_vec)
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
        with dolfinx.io.XDMFFile(self.mesh.comm, dirname + "/" + EXTRACTION_MESH_FILE,
                                 "w") as xdmf:
            xdmf.write_mesh(self.mesh)

        # PETSc matrices

        viewer = PETSc.ViewerHDF5(self.comm).createBinary(
            dirname + "/" + EXTRACTION_VEC_FILE_CTRL_PTS, "w"
        )
        for i in range(0, self.nsd + 1):
            self.generateMesh()
            viewer.view(self.cpFuncs[i].x.petsc_vec)

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
