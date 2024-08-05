import sys

import numpy as np
from petsc4py import PETSc

import dolfinx
import ufl

from dolfinx.fem.petsc import assemble_matrix as petsc_assemble_matrix
from dolfinx.fem.petsc import assemble_vector as petsc_assemble_vector

from tIGArx.calculusUtils import (
    getMetric,
    mappedNormal,
    tIGArxMeasure,
    volumeJacobian,
    surfaceJacobian,
    pinvD,
    getChristoffel,
    cartesianGrad,
    cartesianDiv,
    cartesianCurl,
    CurvilinearTensor,
    curvilinearGrad,
    curvilinearDiv
)

from tIGArx.common import (
    DEFAULT_DO_PERMUTATION,
    INDEX_TYPE,
    FORM_MT,
    EXTRACTION_INFO_FILE,
    EXTRACTION_MESH_FILE,
    EXTRACTION_MAT_FILE,
    EXTRACTION_ZERO_DOFS_FILE,
    EXTRACTION_VEC_FILE_CTRL_PTS,
    EXTRACTION_H5_CONTROL_FUNC_NAME,
    EXTRACTION_MAT_FILE_CTRL,
    worldcomm,
    mpirank, DEFAULT_LINSOLVER_REL_TOL, DEFAULT_LINSOLVER_ABS_TOL,
    DEFAULT_LINSOLVER_MAX_ITERS,
)

from tIGArx.ExtractionGenerator import AbstractExtractionGenerator
from tIGArx.utils import createElementType, createFunctionSpace, \
    createVectorElementType, multTranspose


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
        if mesh is None:
            with dolfinx.io.XDMFFile(self.comm,
                                     dirname + "/" + EXTRACTION_MESH_FILE,
                                     "r") as xdmf:
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
        if F is None:
            F = self.F
        return cartesianGrad(f, F)

    def div(self, f, F=None):
        """
        Cartesian divergence of ``f`` w.r.t. physical coordinates.
        Optional argument ``F``
        can be used to take the gradient assuming a different mapping from
        parametric to physical space.  (Default is ``self.F``.)
        """
        if F is None:
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
        if F is None:
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

    def solveLinearVariationalProblem(self, residualForm, u, applyBCs=True,
                                      linsolverOpts=None):
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
            self, residualForm, J, u, referenceError=None, igaDoFs=None,
            linsolverOpts=None
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
        return self.projectScalarOnto(toProject, self.V_control, linsolverOpts,
                                      lumpMass)

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
