import abc

import numpy as np
import scipy as sp

from petsc4py import PETSc

from tIGArx.ExtractionGenerator import AbstractExtractionGenerator
from tIGArx.common import INDEX_TYPE


class AbstractCoordinateChartSpline(AbstractExtractionGenerator):
    """
    This abstraction represents a spline whose parametric
    coordinate system consists of a
    using a single coordinate chart, so coordinates provide a unique set
    of basis functions; this applies to single-patch B-splines, T-splines,
    NURBS, etc., and, with a little creativity, can be stretched to cover
    multi-patch constructions.
    """

    @abc.abstractmethod
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
