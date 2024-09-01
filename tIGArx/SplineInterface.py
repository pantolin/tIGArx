from __future__ import annotations

import abc

import dolfinx
import numpy as np

from tIGArx.common import worldcomm, DEFAULT_PREALLOC


# abstract class representing a scalar basis of functions on a manifold for
# which we assume that each point has unique coordinates.
class AbstractScalarBasis(object):
    """
    Abstraction defining the behavior of a collection of scalar basis
    functions, defined on a manifold for which each point has unique
    coordinates.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getNodesAndEvals(self, xi):
        """
        Given a parametric point ``xi``, return a list of the form

        ``[[index0, N_index0(xi)], [index1,N_index1(xi)], ... ]``

        where ``N_i`` is the ``i``-th basis function.

        Args:
            xi: A numpy array of parametric points.
        """
        return

    def getNodes(self, xi):
        """
        Given a parametric point ``xi``, return a list of the form

        ``[index0, index1, ... ]``

        where ``index_i`` is the index of the ``i``-th basis function.

        Args:
            xi: A numpy array of parametric points.
        """
        return [node[0] for node in self.getNodesAndEvals(xi)]

    @abc.abstractmethod
    def getNcp(self):
        """
        Returns the total number of basis functions.
        """
        return

    @abc.abstractmethod
    def getElement(self, xi):
        """
        Given a parametric point ``xi``, return the element of the basis
        function that is nonzero at that point.
        """
        return

    @abc.abstractmethod
    def getCpDofmap(self, cells: np.ndarray | None = None, block_size=1) -> np.ndarray:
        """
        Returns a numpy array of control point degrees of freedom associated
        with the given cells. If ``cells`` is None, then all control points
        are returned in the Kronecker product order. If a particular order is
        needed, then the ``cells`` argument should be used.

        Args:
            cells: A numpy array of cell indices.
            block_size: The number of values associated with each control point.

        Returns:
            A numpy array of control point degrees of freedom.
        """
        pass

    @abc.abstractmethod
    def getFEDofmap(self, cells: np.ndarray | None = None) -> np.ndarray:
        """
        Returns a numpy array of finite element degrees of freedom associated
        with the given cells. If ``cells`` is None, then all finite element
        degrees of freedom are returned in the Kronecker product order. If a
        particular order is needed, then the ``cells`` argument should be used.

        Args:
            cells: A numpy array of cell indices.
            block_size: The number of values associated with each finite element
                degree of freedom.

        Returns:
            A numpy array of finite element degrees of freedom.
        """
        pass

    def getExtractionOrdering(self, mesh: dolfinx.mesh.Mesh):
        """
        Returns a list of the form ``[cell0, cell1, ...]``, where
        ``cell_i`` is an integer corresponding to the global index
        in the dolfinx ordering of the ``i``-th cell, used in the
        assembly of the extraction matrix. The mesh is provided as
        the topological ordering is used in 3D, but BEWARE OF 2D, it
        is likely that the ordering is not the same as the topological.
        """
        return mesh.topology.original_cell_index

    @abc.abstractmethod
    def generateMesh(self, comm=worldcomm):
        """
        Generates and returns an FE mesh sufficient for extracting this spline
        basis.  The argument ``comm`` indicates what MPI communicator to
        partition the mesh across, defaulting to ``MPI_COMM_WORLD``.
        """
        return

    @abc.abstractmethod
    def getDegree(self) -> int:
        """
        Returns a polynomial degree for FEs that is sufficient for extracting
        this spline basis.
        """
        pass

    @abc.abstractmethod
    def getNumLocalDofs(self, block_size=1) -> np.ndarray:
        """
        Returns the number of local degrees of freedom for this basis. It
        should not be confused with the number of local dofs in the
        Lagrange basis, it can be lower than that. It is a list because
        it turns out that in some cases, the number of local dofs can vary
        depending on the cell. If the number of local dofs is constant, then
        the list should contain only one element.
        """
        pass

    def getAllNodesAndEvals(self, xi_arr):
        """
        Given a numpy array of parametric points ``xi``, return two numpy
        2D tensors, one of indices and one of evaluations, such that
        ``indices[i,j]`` is the index of the ``j``-th basis function
        evaluated at the ``i``-th point, and ``evals[i,j]`` is the
        corresponding evaluation.
        """
        # Here is a dirty implementation, as straightforward as possible.

        nodes_and_evals_list = [self.getNodesAndEvals(xi) for xi in xi_arr]

        np_arr = np.array(nodes_and_evals_list)

        return np.array(np_arr[:, :, 0], dtype=np.int32), np_arr[:, :, 1]

    @abc.abstractmethod
    def get_lagrange_extraction_operators(self) -> list[np.ndarray]:
        """
        Returns the extraction operators which are used to map between
        the spline basis and the Lagrange basis. There are tensor
        product elements, thus one can return a list of operators. Note
        that the operators need to be a 3D numpy array, where the first
        index is the cell, the second index is the local dof of the
        Lagrange element, and the third index is the local dof of the
        spline element. If the number of local dofs is not constant or
        the tensor product cannot be exploited, then the list contains
        one operator per list element, but the list elements have to be
        3D numpy arrays as well. The first index is a dummy, only there
        to facilitate the structure (numba does not support list of lists).
        The one constraint of this is that one cannot have tensor product
        elements with a varying number of dofs.
        """
        pass

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

    def getSideDofs(self, direction, side, layers=1):
        """
        Returns a list of degrees of freedom on the side of the domain
        specified by the arguments ``direction`` and ``side``.  The argument
        ``layers`` specifies how many layers of degrees of freedom to return.
        """
        pass

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

    def getCSRPrealloc(self, block_size=1) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a pair of numpy arrays, the first of which is the index
        pointer array for a CSR matrix, and the second of which is the
        column index array for a CSR matrix. The pre-allocation needs to
        be exact for the CSR matrix.

        Args:
            block_size: The number of values associated with each degree of freedom.

        Returns:
            A pair of numpy arrays, the first of which is the index pointer
            array and the second of which is the column index array.
        """
        pass


# interface needed for a control mesh with a coordinate chart
class AbstractControlMesh(object):
    """
    Abstraction representing the behavior of a control mesh, i.e., a mapping
    from parametric to physical space.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def getHomogeneousCoordinate(self, node, direction) -> float:
        """
        Returns the ``direction``-th homogeneous component of the control
        point with index ``node``.
        """
        pass

    @abc.abstractmethod
    def get_all_control_points(self) -> np.ndarray:
        """
        Returns a list of all control points in the control mesh.
        """
        pass

    @abc.abstractmethod
    def getScalarSpline(self) -> AbstractScalarBasis:
        """
        Returns the instance of ``AbstractScalarBasis`` that represents
        each homogeneous component of the control mapping.
        """
        pass

    @abc.abstractmethod
    def getNsd(self) -> int:
        """
        Returns the dimension of physical space.
        """
        pass
