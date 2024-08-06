import abc
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
        """
        return

    @abc.abstractmethod
    def getNcp(self):
        """
        Returns the total number of basis functions.
        """
        return

    @abc.abstractmethod
    def generateMesh(self, comm=worldcomm):
        """
        Generates and returns an FE mesh sufficient for extracting this spline
        basis.  The argument ``comm`` indicates what MPI communicator to
        partition the mesh across, defaulting to ``MPI_COMM_WORLD``.
        """
        return

    @abc.abstractmethod
    def getDegree(self):
        """
        Returns a polynomial degree for FEs that is sufficient for extracting
        this spline basis.
        """
        return

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
