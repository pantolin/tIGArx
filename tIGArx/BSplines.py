"""
The ``BSplines`` module 
-----------------------
provides a self-contained implementation of B-splines
that can be used to generate simple sets of extraction data for 
rectangular domains.
"""

import math

import numpy as np
import numba as nb

import dolfinx
from dolfinx import default_real_type

from tIGArx.common import INDEX_TYPE, worldcomm
from tIGArx.SplineInterface import AbstractScalarBasis, AbstractControlMesh
from tIGArx.utils import interleave_and_expand, interleave_and_expand_numba


def uniform_knots(p, start, end, n_elem, periodic=False, continuity_drop=0):
    """
    Helper function to generate a uniform open knot vector of degree ``p`` with
    ``n_elem`` elements.  If ``periodic``, end knots are not repeated.
    Otherwise, they are repeated ``p+1`` times for an open knot vector.
    Optionally, a drop in continuity can be set through the parameter
    ``continuityDrop`` (i.e., interior knots would have multiplicity
    ``continuityDrop+1``).  Negative continuity (i.e., a discontinuous
    B-spline) is not supported, so ``continuityDrop`` must be less than ``p``.

    Args:
        p: int, Polynomial degree
        start: float, Start of domain
        end: float, End of domain
        n_elem: int, Number of elements
        periodic: bool, Periodic knot vector
        continuity_drop: int, Continuity drop

    Returns:
        k_vec: numpy array of floats
    """
    if continuity_drop >= p:
        print("ERROR: Continuity drop too high for spline degree.")
        exit()
    retval = []
    if not periodic:
        for i in range(0, p - continuity_drop):
            retval += [
                start,
            ]
    h = (end - start) / float(n_elem)
    for i in range(0, n_elem + 1):
        for j in range(0, continuity_drop + 1):
            retval += [
                start + float(i) * h,
            ]
    if not periodic:
        for i in range(0, p - continuity_drop):
            retval += [
                end,
            ]
    return retval


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def compute_local_bezier_extraction_operators(k_vec, p):
    """
    Compute the local extraction operators for a B-spline basis with knot
    vector ``k_vec`` and polynomial degree ``p``.  The extraction operators
    are used to extract the spline basis functions from the Bernstein
    basis functions - transforming the basis functions from the Bezier
    space to the B-spline space.

    Args:
        k_vec: numpy array of floats, Knot vector
        p: int, Polynomial degree

    Returns:
        c: numpy array of shape (n, p + 1, p + 1), Extraction operators
    """

    m = len(k_vec)

    # Count the multiplicity of all the knots
    knot_positions = []
    for i in range(1, m):
        if k_vec[i] == k_vec[i - 1]:
            continue
        knot_positions.append(i)
    # Add the last knot as placeholder, assumed to be open
    knot_positions.append(m)

    c = np.zeros((len(knot_positions) - 1, p + 1, p + 1))
    c[0, :, :] = np.eye(p + 1)

    for n in range(0, len(knot_positions) - 1):
        ind_start = knot_positions[n] - 1
        ind_end = knot_positions[n + 1] - 1

        mult = ind_end - ind_start

        if mult < p:
            c[n + 1, :, :] = np.eye(p + 1)  # Initialize the next extraction operator

            # Compute the alphas
            numer = k_vec[ind_end] - k_vec[ind_start]
            alphas = np.zeros(p - mult)

            for j in range(p, mult, -1):
                alphas[j - mult - 1] = numer / (k_vec[ind_start + j] - k_vec[ind_start])

            # Update the matrix coefficients for r new knots
            r = p - mult
            for j in range(1, r + 1):
                s = mult + j

                for k in range(p, s - 1, -1):
                    alpha = alphas[k - s]
                    c[n, :, k] = alpha * c[n, :, k] + (1.0 - alpha) * c[n, :, k - 1]

                if ind_end < m:
                    # The range : is exclusive, so we need to add 1 to the end
                    c[n + 1, (r - j):(r + 1), r - j] = c[n, (p - j):(p + 1), p]

    return c


# need a custom eps for checking knots; dolfin_eps is too small and doesn't
# reliably catch repeated knots
KNOT_NEAR_EPS = 10.0 * np.finfo(default_real_type).eps


# cProfile identified basis function evaluation as a bottleneck in the
# preprocessing, so i've moved it into an inline C++ routine, using
# dolfinx's extension module compilation
basisFuncsCXXString = """
#include <dolfinx/common/Array.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

namespace dolfinx {

int flatIndex(int i, int j, int N){
  return i*N + j;
}

//void basisFuncsInner(const Array<double> &ghostKnots,
//                     int nGhost,
//                     double u,
//                     int pl,
//                     int i,
//                     const Array<double> &ndu,
//                     const Array<double> &left,
//                     const Array<double> &right,
//                     const Array<double> &ders){

typedef py::array_t<double, py::array::c_style | py::array::forcecast> 
    npArray;

void basisFuncsInner(npArray ghostKnots,
                     int nGhost,
                     double u,
                     int pl,
                     int i,
                     npArray ndu,
                     npArray left,
                     npArray right,
                     npArray ders){

    // Technically results in un-defined behavior:
    //Array<double> *ghostKnotsp = const_cast<Array<double>*>(&ghostKnots);
    //Array<double> *ndup = const_cast<Array<double>*>(&ndu);
    //Array<double> *leftp = const_cast<Array<double>*>(&left);
    //Array<double> *rightp = const_cast<Array<double>*>(&right);
    //Array<double> *dersp = const_cast<Array<double>*>(&ders);

    auto ghostKnotsb = ghostKnots.request();
    auto ndub = ndu.request();
    auto leftb = left.request();
    auto rightb = right.request();
    auto dersb = ders.request();

    double *ghostKnotsp = (double *)ghostKnotsb.ptr;
    double *ndup = (double *)ndub.ptr;
    double *leftp = (double *)leftb.ptr;
    double *rightp = (double *)rightb.ptr;
    double *dersp = (double *)dersb.ptr;

    int N = pl+1;
    ndup[flatIndex(0,0,N)] = 1.0;
    for(int j=1; j<pl+1; j++){
        leftp[j] = u - ghostKnotsp[i-j+nGhost];
        rightp[j] = ghostKnotsp[i+j-1+nGhost]-u;
        double saved = 0.0;
        for(int r=0; r<j; r++){
            ndup[flatIndex(j,r,N)] = rightp[r+1] + leftp[j-r];
            double temp = ndup[flatIndex(r,j-1,N)]
                          /ndup[flatIndex(j,r,N)];
            ndup[flatIndex(r,j,N)] = saved + rightp[r+1]*temp;
            saved = leftp[j-r]*temp;
        } // r
        ndup[flatIndex(j,j,N)] = saved;
    } // j
    for(int j=0; j<pl+1; j++){
        dersp[j] = ndup[flatIndex(j,pl,N)];
    } // j
}

PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("basisFuncsInner",&basisFuncsInner,"");
}
}
"""

# basisFuncsCXXModule = compile_extension_module(basisFuncsCXXString,
#                                               cppargs='-g -O2')
# FIXME
# basisFuncsCXXModule = dolfinx.jit.pybind11jit.compile_cpp_code(
#     basisFuncsCXXString)

# function to eval B-spline basis functions (for internal use)
@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def basisFuncsInner(ghostKnots, nGhost, u, pl, i, ndu, left, right, ders):

    # TODO: Fix C++ module for 2018.1, and restore this (or equivalent)
    # call to a compiled routine.
    #
    # basisFuncsCXXModule.basisFuncsInner(ghostKnots,nGhost,u,pl,i,
    #                                    ndu.flatten(),
    #                                    left,right,ders)

    # FIXME -- start: Probably the simplest is to use numba here.
    # TODO: Check if numba is good enough.
    # basisFuncsCXXModule.basisFuncsInner(
    #     ghostKnots, nGhost, u, pl, i, ndu.flatten(), left, right, ders
    # )

    # Backup Python implementation:
    #
    ndu[0, 0] = 1.0
    for j in range(1, pl+1):
        left[j] = u - ghostKnots[i-j+nGhost]
        right[j] = ghostKnots[i+j-1+nGhost]-u
        saved = 0.0
        for r in range(0, j):
            ndu[j, r] = right[r+1] + left[j-r]
            temp = ndu[r, j-1]/ndu[j, r]
            ndu[r, j] = saved + right[r+1]*temp
            saved = left[j-r]*temp
        ndu[j, j] = saved
    for j in range(0, pl+1):
        ders[j] = ndu[j, pl]
    # FIXME -- end


class BSpline1(object):
    """
    Scalar univariate B-spline; this is used construct tensor products, with a
    univariate "tensor product" as a special case; therefore this does not
    implement the ``AbstractScalarBasis`` interface, even though you might
    initially think that it should.
    """

    # create from degree, knot data
    def __init__(self, p, knots):
        """
        Creates a univariate B-spline from a degree, ``p``, and an ordered
        collection of ``knots``.
        """
        self.p = p
        self.knots = np.array(knots)
        self.computeNel()

        # needed for mesh generation
        self.uniqueKnots = np.zeros(self.nel + 1)
        self.multiplicities = np.zeros(self.nel + 1, dtype=INDEX_TYPE)
        ct = -1
        last_knot = None
        for i in range(0, len(self.knots)):
            if last_knot is None or (
                not math.isclose(
                    self.knots[i], last_knot, abs_tol=KNOT_NEAR_EPS, rel_tol=0.0)
            ):
                ct += 1
                self.uniqueKnots[ct] = knots[i]
            last_knot = knots[i]
            self.multiplicities[ct] += 1
        self.ncp = self.computeNcp()

        # knot array with ghosts, for optimized access
        self.nGhost = self.p + 1
        # self.ghostKnots = []
        # for i in range(-self.nGhost,len(self.knots)+self.nGhost):
        #    self.ghostKnots += [self.getKnot(i),]
        # self.ghostKnots = array(self.ghostKnots)
        self.ghostKnots = self.computeGhostKnots()

    def computeGhostKnots(self):
        """
        Pre-compute ghost knots and return as a numpy array.  (Primarily
        intended for internal use.)
        """
        ghostKnots = []
        for i in range(-self.nGhost, len(self.knots) + self.nGhost):
            ghostKnots += [
                self.getKnot(i),
            ]
        return np.array(ghostKnots)

    def normalizeKnotVector(self):
        """
        Re-scales knot vector to be from 0 to 1.
        """
        L = self.knots[-1] - self.knots[0]
        self.knots = (self.knots - self.knots[0]) / L
        self.uniqueKnots = (self.uniqueKnots - self.uniqueKnots[0]) / L
        self.ghostKnots = self.computeGhostKnots()

    # if any non-end knot is repeated more than p time, then the B-spline
    # is discontinuous
    def isDiscontinuous(self):
        """
        Returns a Boolean indicating whether or not the B-spline is
        discontinuous.
        """
        for i in range(1, len(self.uniqueKnots) - 1):
            if self.multiplicities[i] > self.p:
                return True
        return False

    def computeNel(self):
        """
        Returns the number of non-degenerate knot spans.
        """
        self.nel = 0
        for i in range(1, len(self.knots)):
            if not math.isclose(self.knots[i], self.knots[i - 1], abs_tol=KNOT_NEAR_EPS, rel_tol=0.0):
                self.nel += 1

    def getKnot(self, i):
        """
        return a knot, with a (possibly) out-of-range index ``i``.  If ``i``
        is out of range, ghost knots are conjured by looking at the other
        end of the vector.
        Assumes that the first and last unique knots are duplicates with the
        same multiplicity.
        """
        if i < 0:
            ii = len(self.knots) - self.multiplicities[-1] + i
            return self.knots[0] - (self.knots[-1] - self.knots[ii])
        elif i >= len(self.knots):
            ii = i - len(self.knots) + self.multiplicities[0]
            return self.knots[-1] + (self.knots[ii] - self.knots[0])
        else:
            return self.knots[i]

    def greville(self, i):
        """
        Returns the Greville parameter associated with the
        ``i``-th control point.
        """
        retval = 0.0
        for j in range(i, i + self.p):
            retval += self.getKnot(j + 1)
        retval /= float(self.p)
        return retval

    def computeNcp(self):
        """
        Computes and returns the number of control points.
        """
        return len(self.knots) - self.multiplicities[0]

    def getNcp(self):
        """
        Returns the number of control points.
        """
        return self.ncp

    def getKnotSpan(self, u):
        """
        Given parameter ``u``, return the index of the knot span in which
        ``u`` falls.  (Numbering includes degenerate knot spans.)
        """

        # placeholder linear search
        # span = 0
        # nspans = len(self.knots)-1
        # for i in range(0,nspans):
        #    span = i
        #    if(u<self.knots[i+1]+np.finfo(default_real_type).eps):
        #        break

        # from docs: should be index of "rightmost value less than x"
        nspans = len(self.knots) - 1
        # span = bisect.bisect_left(self.knots,u)-1
        span = np.searchsorted(self.knots, u) - 1

        if span < self.multiplicities[0] - 1:
            span = self.multiplicities[0] - 1
        if span > nspans - (self.multiplicities[-1] - 1) - 1:
            span = nspans - (self.multiplicities[-1] - 1) - 1
        return span

    def getNodes(self, u):
        """
        Given a parameter ``u``, return a list of the indices of B-spline
        basis functions whose supports contain ``u``.
        """
        nodes = []
        knotSpan = self.getKnotSpan(u)
        for i in range(knotSpan - self.p, knotSpan + 1):
            nodes += [
                i % self.getNcp(),
            ]
        return nodes

    def basisFuncs(self, knotSpan, u):
        """
        Return a list of the ``p+1`` nonzero basis functions evaluated at
        the parameter ``u`` in the knot span with index ``knotSpan``.
        """
        pl = self.p
        # u_knotl = self.knots
        i = knotSpan + 1
        ndu = np.zeros((pl + 1, pl + 1))
        left = np.zeros(pl + 1)
        right = np.zeros(pl + 1)
        ders = np.zeros(pl + 1)

        basisFuncsInner(self.ghostKnots, self.nGhost, u,
                        pl, i, ndu, left, right, ders)

        # ndu[0,0] = 1.0
        # for j in range(1,pl+1):
        #    left[j] = u - self.getKnot(i-j) #u_knotl[i-j]
        #    right[j] = self.getKnot(i+j-1)-u #u_knotl[i+j-1]-u
        #    saved = 0.0
        #    for r in range(0,j):
        #        ndu[j,r] = right[r+1] + left[j-r]
        #        temp = ndu[r,j-1]/ndu[j,r]
        #        ndu[r,j] = saved + right[r+1]*temp
        #        saved = left[j-r]*temp
        #    ndu[j,j] = saved
        # for j in range(0,pl+1):
        #    ders[j] = ndu[j,pl]

        return ders

    def compute_local_lagrange_extraction_operator(self, order=None):
        """
        Compute the local Lagrange extraction operator for the B-spline basis.
        By default the order will be the same as the polynomial degree of the
        B-spline basis. However, optionally it can be higher to allow for
        higher-order extraction. The matrices in this case are no longer
        square but of shape (self.p + 1, order + 1).

        Args:
            order: int, Order of the extraction operator

        Returns:
            3D numpy tensor, local extraction operators of shape
            (self.nel, self.p + 1, order + 1) where order = self.p if
            not specified
        """
        # Right now periodic B-splines are not supported
        assert self.multiplicities[0] == self.multiplicities[-1] == self.p + 1

        p = self.p
        if order is None:
            order = p
        else:
            if order < p:
                print("ERROR: Order of extraction operator must be at "
                      "least the same as the polynomial degree.")
                exit()

        # C-order is vital for numba
        operators = np.zeros((self.nel, p + 1, order + 1),
                             dtype=default_real_type, order='C')

        value = self.basisFuncs(p, self.uniqueKnots[0])
        operators[0, :, 0] = np.array(value, order='C')

        start_func = 0

        for i in range(0, self.nel):
            start_knot = self.uniqueKnots[i]
            end_knot = self.uniqueKnots[i + 1]

            # Subdivide the interval into p sub-intervals
            h = (end_knot - start_knot) / order
            m = self.multiplicities[i + 1]
            # Compute the extraction operator for each sub-interval
            for j in range(1, order + 1):
                value = self.basisFuncs(start_func + p, start_knot + j * h)
                operators[i, :, j] = np.array(value, order='C')

            if i < self.nel - 1:
                operators[i + 1, 0:(p + 1 - m), 0] = operators[i, m:(p + 1), order]

            start_func += m

        return operators

        # This is for compatibility purposes with general extraction case
        # where each element can have a different number of local dofs
        # operator_list = []
        # for i in range(self.nel):
        #     operator_list.append(np.ascontiguousarray(operators[i]))


    def compute_start_indices(self):
        """
        Compute the start indices of the control points in the global
        control point vector.
        """
        start_indices = np.zeros(self.nel, dtype=np.int32)

        start_indices[0] = 0
        for i in range(1, self.nel):
            start_indices[i] = start_indices[i - 1] + self.multiplicities[i]

        return start_indices

    def compute_dofs(self):
        """
        Compute the global degrees of freedom of the B-spline basis.
        """
        start_indices = self.compute_start_indices()

        dofs_per_span = (start_indices[:, np.newaxis]
                         + np.arange(self.p + 1)[np.newaxis, :])

        return dofs_per_span

    def interacting_basis_functions(self):
        """
        Returns a list of lists, where the i-th list contains the
        global degrees of freedom of basis functions that interact
        with the i-th basis function
        """
        dofs = self.compute_dofs()
        func_support = [[] for _ in range(self.ncp)]

        for interval, funcs_on_interval in enumerate(dofs):
            for func in funcs_on_interval:
                func_support[func].append(interval)

        interaction_list = []

        for i in range(self.ncp):
            local_list = []

            for interval in func_support[i]:
                local_list += list(dofs[interval, :])

            interaction_list.append(
                np.array(
                    np.sort(np.unique(local_list)),
                    dtype=np.int32
                )
            )

        return interaction_list


# utility functions for indexing (mainly for internal library use)
def ij2dof(i, j, M):
    return j * M + i


def ijk2dof(i, j, k, M, N):
    return k * (M * N) + j * M + i


def dof2ij(dof, M):
    i = dof % M
    j = dof // M
    return (i, j)


def dof2ijk(dof, M, N):
    ij = dof % (M * N)
    i = ij % M
    j = ij // M
    k = dof // (M * N)
    return (i, j, k)


# Use BSpline1 instances to store info about each dimension.  No point in going
# higher than 3, since FEniCS only generates meshes up to dimension 3...
# -> Suggestion: dimensions 1, 2, and 3 are sufficiently different that they
# should probably be implemented as separate classes.
class BSpline(AbstractScalarBasis):
    """
    Class implementing the ``AbstractScalarBasis`` interface, to represent
    a uni-, bi-, or tri-variate B-spline.
    """
    def __init__(self, degrees, kvecs, overRefine=0):
        """
        Create a ``BSpline`` with degrees in each direction given by the
        sequence ``degrees``, and knot vectors given by the list of
        sequences ``kvecs``.  The optional parameter ``overRefine``
        indicates how many levels of refinement to apply beyond what is
        needed to represent the spline functions; choosing a value greater
        than the default of zero may be useful for
        integrating functions with fine-scale features.

        NOTE: Over-refinement is only supported with simplicial elements.
        """
        self.nvar = len(degrees)
        if self.nvar > 3 or self.nvar < 1:
            print("ERROR: Unsupported parametric dimension.")
            exit()

        self.splines: list[BSpline1] = []
        for i in range(0, self.nvar):
            self.splines += [
                BSpline1(degrees[i], kvecs[i]),
            ]

        # TODO - remove this constraint and enable periodic splines
        for s in self.splines:
            if s.multiplicities[0] != s.multiplicities[-1] != s.p + 1:
                print("ERROR: Only open knot vectors are supported.")
                exit()

        self.overRefine = overRefine
        self.ncp = self.computeNcp()
        self.nel = self.computeNel()

    def normalizeKnotVectors(self):
        """
        Scale knot vectors in all directions to (0,1).
        """
        for s in self.splines:
            s.normalizeKnotVector()

    # def getParametricDimension(self):
    #    return self.nvar

    def needsDG(self):
        """
        Returns a Boolean, indicating whether or not the extraction requires
        DG element (due to the function space being discontinuous somewhere).
        """
        for i in range(0, self.nvar):
            if self.splines[i].isDiscontinuous():
                return True
        return False

    # non-default implementation, optimized for B-splines
    def getPrealloc(self):
        totalFuncs = 1
        for spline in self.splines:
            # a 1d b-spline should have p+1 active basis functions at any given
            # point; however, when eval-ing at boundaries of knot spans,
            # may pick up at most 2 extra basis functions due to epsilons.
            #
            # TODO: find a way to automatically ignore those extra nearly-zero
            # basis functions.
            # totalFuncs *= (spline.p+1 +2)
            # TODO: Shouldn't the pessimistic estimate be 2*p+1 per spline?
            totalFuncs *= spline.p + 1
        return totalFuncs

    def is_tensor_product_basis(self) -> bool:
        return True

    def getNodesAndEvals(self, xi):

        if self.nvar == 1:
            u = xi[0]
            span = self.splines[0].getKnotSpan(u)
            nodes = self.splines[0].getNodes(u)
            ders = self.splines[0].basisFuncs(span, u)
            retval = []
            for i in range(0, len(nodes)):
                retval += [
                    [nodes[i], ders[i]],
                ]
            return retval
        elif self.nvar == 2:
            u = xi[0]
            v = xi[1]
            uspline = self.splines[0]
            vspline = self.splines[1]
            spanu = uspline.getKnotSpan(u)
            spanv = vspline.getKnotSpan(v)
            nodesu = uspline.getNodes(u)
            nodesv = vspline.getNodes(v)
            dersu = uspline.basisFuncs(spanu, u)
            dersv = vspline.basisFuncs(spanv, v)
            retval = []
            for i in range(0, len(nodesu)):
                for j in range(0, len(nodesv)):
                    retval += [
                        [
                            ij2dof(nodesu[i], nodesv[j], uspline.getNcp()),
                            dersu[i] * dersv[j],
                        ],
                    ]
            return retval
        else:
            u = xi[0]
            v = xi[1]
            w = xi[2]
            uspline = self.splines[0]
            vspline = self.splines[1]
            wspline = self.splines[2]
            spanu = uspline.getKnotSpan(u)
            spanv = vspline.getKnotSpan(v)
            spanw = wspline.getKnotSpan(w)
            nodesu = uspline.getNodes(u)
            nodesv = vspline.getNodes(v)
            nodesw = wspline.getNodes(w)
            dersu = uspline.basisFuncs(spanu, u)
            dersv = vspline.basisFuncs(spanv, v)
            dersw = wspline.basisFuncs(spanw, w)
            retval = []
            for i in range(0, len(nodesu)):
                for j in range(0, len(nodesv)):
                    for k in range(0, len(nodesw)):
                        retval += [
                            [
                                ijk2dof(
                                    nodesu[i],
                                    nodesv[j],
                                    nodesw[k],
                                    uspline.getNcp(),
                                    vspline.getNcp(),
                                ),
                                dersu[i] * dersv[j] * dersw[k],
                            ],
                        ]
            return retval

    def getNodes(self, xi):
        ret_nodes: np.ndarray

        if self.nvar == 1:
            u = xi[0]
            nodes = self.splines[0].getNodes(u)
            ret_nodes = np.array(nodes, dtype=np.int32)

        elif self.nvar == 2:
            u = xi[0]
            v = xi[1]

            uspline = self.splines[0]
            vspline = self.splines[1]

            nodesu = np.array(uspline.getNodes(u))
            nodesv = np.array(vspline.getNodes(v))
            ret_nodes = np.zeros(len(nodesu) * len(nodesv), dtype=np.int32)

            for j in range(0, len(nodesv)):
                for i in range(0, len(nodesu)):
                    ret_nodes[j * len(nodesu) + i] = (
                        nodesv[j] * uspline.getNcp() + nodesu[i]
                    )

        else:
            u = xi[0]
            v = xi[1]
            w = xi[2]

            uspline = self.splines[0]
            vspline = self.splines[1]
            wspline = self.splines[2]

            nodesu = np.array(uspline.getNodes(u))
            nodesv = np.array(vspline.getNodes(v))
            nodesw = np.array(wspline.getNodes(w))
            ret_nodes = np.zeros(len(nodesu) * len(nodesv) * len(nodesw), dtype=np.int32)

            n_u = len(nodesu)
            n_v = len(nodesv)
            n_w = len(nodesw)

            for k in range(0, n_w):
                for j in range(0, n_v):
                    for i in range(0, n_u):
                        ret_nodes[k * n_v * n_u + j * n_u + i] = (
                            nodesw[k] * (uspline.getNcp() * vspline.getNcp())
                            + nodesv[j] * uspline.getNcp()
                            + nodesu[i]
                        )

        return ret_nodes

    def generateMesh(self, comm=worldcomm):
        if self.nvar == 1:
            spline = self.splines[0]
            mesh = dolfinx.mesh.create_unit_interval(comm, spline.nel)

            x = mesh.geometry.x
            for i in range(0, len(x)):
                knotIndex = int(round(x[i, 0] * float(spline.nel)))
                x[i, 0] = spline.uniqueKnots[knotIndex]
            # return mesh

        elif self.nvar == 2:
            uspline = self.splines[0]
            vspline = self.splines[1]
            cellType = dolfinx.mesh.CellType.quadrilateral
            mesh = dolfinx.mesh.create_unit_square(
                comm, uspline.nel, vspline.nel, cellType
            )

            x = mesh.geometry.x
            for i in range(0, len(x)):
                # uknotIndex = int(round(x[i,0]))
                # vknotIndex = int(round(x[i,1]))
                uknotIndex = int(round(x[i, 0] * float(uspline.nel)))
                vknotIndex = int(round(x[i, 1] * float(vspline.nel)))

                x[i, 0] = uspline.uniqueKnots[uknotIndex]
                x[i, 1] = vspline.uniqueKnots[vknotIndex]
            # return mesh
        else:
            uspline = self.splines[0]
            vspline = self.splines[1]
            wspline = self.splines[2]
            cellType = dolfinx.mesh.CellType.hexahedron
            mesh = dolfinx.mesh.create_unit_cube(
                comm, uspline.nel, vspline.nel, wspline.nel, cellType
            )

            x = mesh.geometry.x
            for i in range(0, len(x)):
                # uknotIndex = int(round(x[i,0]))
                # vknotIndex = int(round(x[i,1]))
                # wknotIndex = int(round(x[i,2]))
                uknotIndex = int(round(x[i, 0] * float(uspline.nel)))
                vknotIndex = int(round(x[i, 1] * float(vspline.nel)))
                wknotIndex = int(round(x[i, 2] * float(wspline.nel)))

                x[i, 0] = uspline.uniqueKnots[uknotIndex]
                x[i, 1] = vspline.uniqueKnots[vknotIndex]
                x[i, 2] = wspline.uniqueKnots[wknotIndex]

        # Apply any over-refinement specified:
        for i in range(0, self.overRefine):
            mesh = dolfinx.mesh.refine(mesh)
        return mesh

    def computeNcp(self):
        prod = 1
        for i in range(0, self.nvar):
            prod *= self.splines[i].getNcp()
        return prod

    def getNcp(self):
        return self.ncp

    def getDegree(self):
        deg = 0
        for i in range(0, self.nvar):
            deg = max(deg, self.splines[i].p)
        return deg

    def getNumLocalDofs(self, block_size=1) -> np.ndarray:
        deg = 1
        for i in range(0, self.nvar):
            deg *= self.splines[i].p + 1
        return np.array([deg * block_size])

    def getElement(self, xi):
        """
        Returns the element index that contains the point ``xi``.
        """
        if self.nvar == 1:
            u = xi[0]
            u_spline = self.splines[0]
            span = u_spline.getKnotSpan(u) - u_spline.multiplicities[0] + 1
            return span
        elif self.nvar == 2:
            u = xi[0]
            v = xi[1]

            u_spline = self.splines[0]
            v_spline = self.splines[1]

            u_span = u_spline.getKnotSpan(u) - u_spline.multiplicities[0] + 1
            v_span = v_spline.getKnotSpan(v) - v_spline.multiplicities[0] + 1
            # The first value is y direction, second x
            return v_span * u_spline.nel + u_span
        else:
            u = xi[0]
            v = xi[1]
            w = xi[2]

            u_spline = self.splines[0]
            v_spline = self.splines[1]
            w_spline = self.splines[2]

            u_span = u_spline.getKnotSpan(u) - u_spline.multiplicities[0] + 1
            v_span = v_spline.getKnotSpan(v) - v_spline.multiplicities[0] + 1
            w_span = w_spline.getKnotSpan(w) - w_spline.multiplicities[0] + 1
            # The first value is z direction, second y, third x
            return (w_span * (u_spline.nel * v_spline.nel)
                    + v_span * u_spline.nel
                    + u_span)

    def getCpDofmap(self, cells: np.ndarray | None = None, block_size=1):
        if cells is None:
            cells = np.arange(self.nel, dtype=np.int32)

        dofs: np.ndarray

        if self.nvar == 1:
            u_indices = self.splines[0].compute_dofs()

            dofs = np.empty(
                (cells.shape[0], (self.splines[0].p + 1) * block_size),
                dtype=np.int32
            )

            for i, cell in enumerate(cells.data):
                temp_dofs = u_indices[cell]
                dofs[i, :] = interleave_and_expand(temp_dofs, block_size)

        elif self.nvar == 2:
            u_spline = self.splines[0]
            v_spline = self.splines[1]

            u_indices = u_spline.compute_dofs()
            v_indices = v_spline.compute_dofs()

            u_nel = u_spline.nel

            dofs = np.empty(
                (
                    cells.shape[0],
                    (u_spline.p + 1) * (v_spline.p + 1) * block_size
                ),
                dtype=np.int32
            )

            @nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
            def fill_dofs_2d(
                    u_indices, v_indices, cells, dofs, u_ncp, u_nel, block_size
            ):
                pos = 0
                for cell in cells:
                    v_span = cell // u_nel
                    u_span = cell % u_nel

                    u_loc = u_indices[u_span]
                    v_loc = v_indices[v_span]

                    n_u = u_loc.size
                    n_v = v_loc.size

                    temp_dofs = np.zeros(n_u * n_v, dtype=np.int32)

                    for j in range(n_v):
                        for i in range(n_u):
                            temp_dofs[j * n_u + i] = v_loc[j] * u_ncp + u_loc[i]

                    dofs[pos, :] = interleave_and_expand_numba(temp_dofs, block_size)
                    pos += 1

            fill_dofs_2d(
                u_indices, v_indices, cells, dofs,
                u_spline.getNcp(), u_nel,
                block_size
            )

            # for i, cell in enumerate(cells.data):
            #     v_span = cell // u_nel
            #     u_span = cell % u_nel
            #
            #     temp_dofs = np.add.outer(
            #         v_indices[v_span] * u_spline.getNcp(),
            #         u_indices[u_span]
            #     ).reshape(-1)
            #
            #     dofs[i, :] = interleave_and_expand(temp_dofs, block_size)

        else:
            u_spline = self.splines[0]
            v_spline = self.splines[1]
            w_spline = self.splines[2]

            u_indices = u_spline.compute_dofs()
            v_indices = v_spline.compute_dofs()
            w_indices = w_spline.compute_dofs()

            u_nel = u_spline.nel
            v_nel = v_spline.nel

            dofs = np.empty(
                (
                    cells.shape[0],
                    (u_spline.p + 1) * (v_spline.p + 1) * (w_spline.p + 1) * block_size
                ),
                dtype=np.int32
            )

            @nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
            def fill_dofs_3d(
                    u_indices, v_indices, w_indices, cells, dofs,
                    u_ncp, v_ncp, u_nel, v_nel, block_size):
                pos = 0
                for cell in cells:
                    w_span = cell // (u_nel * v_nel)
                    v_span = (cell // u_nel) % v_nel
                    u_span = cell % u_nel

                    u_loc = u_indices[u_span]
                    v_loc = v_indices[v_span]
                    w_loc = w_indices[w_span]

                    n_u = u_loc.size
                    n_v = v_loc.size
                    n_w = w_loc.size

                    temp_dofs = np.zeros(n_u * n_v * n_w, dtype=np.int32)

                    for k in range(n_w):
                        for j in range(n_v):
                            for i in range(n_u):
                                temp_dofs[k * n_v * n_u + j * n_u + i] = (
                                    w_loc[k] * (u_ncp * v_ncp)
                                    + v_loc[j] * u_ncp
                                    + u_loc[i]
                                )

                    dofs[pos, :] = interleave_and_expand_numba(temp_dofs, block_size)
                    pos += 1

            fill_dofs_3d(
                u_indices, v_indices, w_indices,
                cells, dofs,
                u_spline.getNcp(), v_spline.getNcp(),
                u_nel, v_nel,
                block_size
            )

            # for i, cell in enumerate(cells.data):
            #     w_span = cell // (u_nel * v_nel)
            #     v_span = (cell // u_nel) % v_nel
            #     u_span = cell % u_nel
            #
            #     temp_dofs = np.add.outer(
            #         w_indices[w_span] * (u_spline.getNcp() * v_spline.getNcp()),
            #         np.add.outer(
            #             v_indices[v_span] * u_spline.getNcp(),
            #             u_indices[u_span]
            #         ).reshape(-1)
            #     ).reshape(-1)
            #
            #     dofs[i, :] = interleave_and_expand(temp_dofs, block_size)

        return dofs

    def getExtractionOrdering(self, mesh: dolfinx.mesh.Mesh):
        """
        Returns the ordering of the control points in the global control point
        vector.
        """
        if self.nvar == 1:
            return np.arange(self.splines[0].nel, dtype=np.int32)

        elif self.nvar == 2:
            # The 2D case is special, as the ordering is not the same as the
            # topo
            n_u = self.splines[0].nel
            n_v = self.splines[1].nel
            cell_matrix = np.arange(n_u * n_v, dtype=np.int32).reshape(n_v, n_u)

            ordering = np.empty(n_u * n_v, dtype=np.int32)
            index = 0

            # Iterate over all possible sums of indices from 0 to M+N-2
            for s in range(n_v + n_u - 1):
                # Give the y-axis index priority
                for i in range(min(n_v - 1, s), max(-1, s - n_u), -1):
                    j = s - i
                    if j < n_u:
                        ordering[index] = cell_matrix[i, j]
                        index += 1

            return ordering

        else:
            # In 3D the actual ordering corresponds to the topological one
            # It could be done in a similar fashion as for 2D, but the
            # traversal is complicated, and corresponds to the established
            # ordering in the mesh.
            return mesh.topology.original_cell_index

    def getCSRPrealloc(self, block_size=1) -> tuple[np.ndarray, np.ndarray]:
        interacting: list[np.ndarray] = []

        if self.nvar == 1:
            u_inter = self.splines[0].interacting_basis_functions()
            for i in range(self.ncp):
                temp = interleave_and_expand(u_inter[i], block_size)
                for _ in range(block_size):
                    interacting.append(np.array(temp, dtype=np.int32))

        elif self.nvar == 2:
            u_spline = self.splines[0]
            v_spline = self.splines[1]

            u_inter = nb.typed.List(u_spline.interacting_basis_functions())
            v_inter = nb.typed.List(v_spline.interacting_basis_functions())

            @nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
            def get_interacting_2d(u_inter, v_inter, u_ncp, v_ncp, block_size):
                interacting = []
                for j in range(v_ncp):
                    for i in range(u_ncp):
                        u_loc = u_inter[i]
                        v_loc = v_inter[j]

                        n_u = u_loc.size
                        n_v = v_loc.size

                        temp = np.zeros(n_u * n_v, dtype=np.int32)

                        for jj in range(n_v):
                            for ii in range(n_u):
                                temp[jj * n_u + ii] = np.int32(
                                    v_loc[jj] * u_ncp + u_loc[ii]
                                )

                        interleaved = interleave_and_expand_numba(temp, block_size)
                        for _ in range(block_size):
                            interacting.append(interleaved)

                return interacting

            interacting = get_interacting_2d(
                u_inter, v_inter,
                u_spline.getNcp(), v_spline.getNcp(),
                block_size
            )

            # for j in range(v_spline.getNcp()):
            #     for i in range(u_spline.getNcp()):
            #         temp = interleave_and_expand(
            #             np.add.outer(
            #                 v_inter[j] * u_spline.getNcp(),
            #                 u_inter[i]
            #             ).reshape(-1),
            #             block_size
            #         )
            #         for _ in range(block_size):
            #             interacting.append(np.array(temp, dtype=np.int32))

        else:
            u_spline = self.splines[0]
            v_spline = self.splines[1]
            w_spline = self.splines[2]

            u_inter = nb.typed.List(u_spline.interacting_basis_functions())
            v_inter = nb.typed.List(v_spline.interacting_basis_functions())
            w_inter = nb.typed.List(w_spline.interacting_basis_functions())

            @nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
            def get_interacting_3d(
                    u_inter, v_inter, w_inter,
                    u_ncp, v_ncp, w_ncp, block_size
            ):
                interacting = []
                for k in range(w_ncp):
                    for j in range(v_ncp):
                        for i in range(u_ncp):
                            u_loc = u_inter[i]
                            v_loc = v_inter[j]
                            w_loc = w_inter[k]

                            n_u = u_loc.size
                            n_v = v_loc.size
                            n_w = w_loc.size

                            temp = np.zeros(n_u * n_v * n_w, dtype=np.int32)

                            for l in range(n_w):
                                for m in range(n_v):
                                    for n in range(n_u):
                                        temp[l * n_v * n_u + m * n_u + n] = np.int32(
                                            w_loc[l] * (u_ncp * v_ncp)
                                            + v_loc[m] * u_ncp
                                            + u_loc[n]
                                        )

                            interleaved = interleave_and_expand_numba(temp, block_size)
                            for _ in range(block_size):
                                interacting.append(interleaved)

                return interacting

            interacting = get_interacting_3d(
                u_inter, v_inter, w_inter,
                u_spline.getNcp(), v_spline.getNcp(), w_spline.getNcp(),
                block_size
            )

            # for k in range(w_spline.getNcp()):
            #     for j in range(v_spline.getNcp()):
            #         for i in range(u_spline.getNcp()):
            #             temp = interleave_and_expand(
            #                 np.add.outer(
            #                     w_inter[k] * (u_spline.getNcp() * v_spline.getNcp()),
            #                     np.add.outer(
            #                         v_inter[j] * u_spline.getNcp(),
            #                         u_inter[i]
            #                     ).reshape(-1)
            #                 ).reshape(-1),
            #                 block_size
            #             )
            #             for _ in range(block_size):
            #                 interacting.append(np.array(temp, dtype=np.int32))

        # repeat the list interacting block_size times
        # interacting = interacting * block_size

        index_ptr = np.zeros(self.ncp * block_size + 1, dtype=np.int32)
        index_ptr[0] = 0

        for i in range(self.ncp * block_size):
            index_ptr[i + 1] = index_ptr[i] + len(interacting[i])

        return index_ptr, np.concatenate(interacting)

    def get_lagrange_extraction_operators(self) -> nb.typed.List[np.ndarray]:
        """
        Returns a list of local Lagrange extraction operators, one for each
        unique knot span.
        """
        operators = nb.typed.List()
        order = self.getDegree()
        for i in range(0, self.nvar):
            operators.append(
                self.splines[i].compute_local_lagrange_extraction_operator(order=order)
            )
        return operators

    def computeNel(self):
        """
        Returns the number of Bezier elements in the B-spline.
        """
        nel = 1
        for spline in self.splines:
            nel *= spline.nel
        return nel

    def getSideDofs(self, direction, side, layers=1):
        """
        Return the DoFs on a ``side`` (zero or one) that is perpendicular
        to a parametric ``direction`` (0, 1, or 2, capped at
        ``self.nvar-1``, obviously).  Can optionally constrain more than
        one layer of control points (e.g., for strongly-enforced clamped BCs
        on Kirchhoff--Love shells) using ``layers`` greater than its
        default value of one.
        """
        offsetSign = 1 - 2 * side
        retval = []
        for absOffset in range(0, layers):
            offset = absOffset * offsetSign
            if side == 0:
                i = 0
            else:
                i = self.splines[direction].getNcp() - 1
            i += offset
            M = self.splines[0].getNcp()
            if self.nvar == 1:
                retval += [
                    i,
                ]
                continue
            N = self.splines[1].getNcp()
            if self.nvar == 2:
                dofs = []
                if direction == 0:
                    for j in range(0, N):
                        dofs += [
                            ij2dof(i, j, M),
                        ]
                elif direction == 1:
                    for j in range(0, M):
                        dofs += [
                            ij2dof(j, i, M),
                        ]
                retval += dofs
                continue
            O = self.splines[2].getNcp()
            if self.nvar == 3:
                dofs = []
                if direction == 0:
                    for j in range(0, N):
                        for k in range(0, O):
                            dofs += [
                                ijk2dof(i, j, k, M, N),
                            ]
                elif direction == 1:
                    for j in range(0, M):
                        for k in range(0, O):
                            dofs += [
                                ijk2dof(j, i, k, M, N),
                            ]
                elif direction == 2:
                    for j in range(0, M):
                        for k in range(0, N):
                            dofs += [
                                ijk2dof(j, k, i, M, N),
                            ]
                retval += dofs
                continue
        return retval


# class MultiBSpline(AbstractScalarBasis):
#     """
#     Several ``BSpline`` instances grouped together.
#     """

#     # TODO: add a mechanism to merge basis functions (analogous to IPER
#     # in the Fortran code) that can be used to merge control points for
#     # equal-order interpolations.  (Should be an integer array of length
#     # self.ncp, with mostly array[i]=i, except for slave nodes.)

#     def __init__(self, splines):
#         """
#         Create a ``MultiBSpline`` from a sequence ``splines`` of individual
#         ``BSpline`` instances.  This sequence is assumed to contain at least
#         one ``BSpline``, and all elements of ``splines`` are assumed to use
#         the same element type, and have the same parametric
#         dimensions as each other.
#         """
#         self.splines = splines
#         self.ncp = self.computeNcp()

#         # normalize all knot vectors to (0,1) for each patch, for easy lookup
#         # of patch index from coordinates
#         for s in self.splines:
#             s.normalizeKnotVectors()

#         # pre-compute DoF index offsets for each patch
#         self.doffsets = []
#         ncp = 0
#         for s in self.splines:
#             self.doffsets += [
#                 ncp,
#             ]
#             ncp += s.getNcp()

#         self.nvar = self.splines[0].nvar
#         self.overRefine = self.splines[0].overRefine
#         self.nPatch = len(self.splines)
#         self.nel = self.computeNel()

#     def computeNel(self):
#         """
#         Returns the number of Bezier elements between all patches.
#         """
#         nel = 0
#         for spline in self.splines:
#             nel += spline.nel
#         return nel

#     # TODO: this should not need to exist
#     def needsDG(self):
#         return False

#     # non-default implementation, optimized for B-splines
#     def getPrealloc(self):
#         return self.splines[0].getPrealloc()

#     def getNodesAndEvals(self, xi):
#         patch = self.patchFromCoordinates(xi)
#         xi_local = self.localParametricCoordinates(xi, patch)
#         localNodesAndEvals = self.splines[patch].getNodesAndEvals(xi_local)
#         retval = []
#         for pair in localNodesAndEvals:
#             retval += [
#                 [self.globalDofIndex(pair[0], patch), pair[1]],
#             ]
#         return retval

#     def patchFromCoordinates(self, xi):
#         return int(xi[0] + 0.5) // 2

#     def globalDofIndex(self, localDofIndex, patchIndex):
#         return self.doffsets[patchIndex] + localDofIndex

#     def localParametricCoordinates(self, xi, patchIndex):
#         retval = xi.copy()
#         retval[0] = xi[0] - 2.0 * float(patchIndex)
#         return retval

#     def generateMesh(self, comm=worldcomm):

#         MESH_FILE_NAME = generateMeshXMLFileName(comm)

#         if comm.Get_rank() == 0:
#             fs = '<?xml version="1.0" encoding="UTF-8"?>' + "\n"
#             fs += '<dolfinx xmlns:dolfinx="http://www.fenics.org/dolfinx/">' + "\n"
#             if self.nvar == 1:
#                 # TODO
#                 print("ERROR: Univariate multipatch not yet supported.")
#                 exit()
#             elif self.nvar == 2:
#                 fs += '<mesh celltype="quadrilateral" dim="2">' + "\n"

#                 # TODO: Do indexing more intelligently, so that elements
#                 # are connected within each patch.  (This will improve
#                 # parallel performance.)

#                 nverts = 4 * self.nel
#                 nel = self.nel
#                 fs += '<vertices size="' + str(nverts) + '">' + "\n"
#                 vertCounter = 0
#                 x00 = 0.0
#                 for patch in range(0, self.nPatch):
#                     spline = self.splines[patch]
#                     uspline = spline.splines[0]
#                     vspline = spline.splines[1]
#                     for i in range(0, uspline.nel):
#                         for j in range(0, vspline.nel):
#                             x0 = repr(x00 + uspline.uniqueKnots[i])
#                             x1 = repr(x00 + uspline.uniqueKnots[i + 1])
#                             y0 = repr(vspline.uniqueKnots[j])
#                             y1 = repr(vspline.uniqueKnots[j + 1])
#                             fs += (
#                                 '<vertex index="'
#                                 + str(vertCounter)
#                                 + '" x="'
#                                 + x0
#                                 + '" y="'
#                                 + y0
#                                 + '"/>'
#                                 + "\n"
#                             )
#                             fs += (
#                                 '<vertex index="'
#                                 + str(vertCounter + 1)
#                                 + '" x="'
#                                 + x1
#                                 + '" y="'
#                                 + y0
#                                 + '"/>'
#                                 + "\n"
#                             )
#                             fs += (
#                                 '<vertex index="'
#                                 + str(vertCounter + 2)
#                                 + '" x="'
#                                 + x0
#                                 + '" y="'
#                                 + y1
#                                 + '"/>'
#                                 + "\n"
#                             )
#                             fs += (
#                                 '<vertex index="'
#                                 + str(vertCounter + 3)
#                                 + '" x="'
#                                 + x1
#                                 + '" y="'
#                                 + y1
#                                 + '"/>'
#                                 + "\n"
#                             )
#                             vertCounter += 4
#                     x00 += 2.0
#                 fs += "</vertices>" + "\n"
#                 fs += '<cells size="' + str(nel) + '">' + "\n"
#                 elCounter = 0
#                 for patch in range(0, self.nPatch):
#                     spline = self.splines[patch]
#                     uspline = spline.splines[0]
#                     vspline = spline.splines[1]
#                     for i in range(0, uspline.nel):
#                         for j in range(0, vspline.nel):
#                             v0 = str(elCounter * 4 + 0)
#                             v1 = str(elCounter * 4 + 1)
#                             v2 = str(elCounter * 4 + 2)
#                             v3 = str(elCounter * 4 + 3)
#                             fs += (
#                                 '<quadrilateral index="'
#                                 + str(elCounter)
#                                 + '" v0="'
#                                 + v0
#                                 + '" v1="'
#                                 + v1
#                                 + '" v2="'
#                                 + v2
#                                 + '" v3="'
#                                 + v3
#                                 + '"/>'
#                                 + "\n"
#                             )
#                             elCounter += 1
#                 fs += "</cells></mesh></dolfinx>"
#             elif self.nvar == 3:
#                 # TODO
#                 print("ERROR: Trivariate multipatch not yet supported.")
#                 exit()
#             else:
#                 # TO NOT DO...
#                 print("ERROR: Unsupported parametric dimension: " + str(self.nvar))
#                 exit()
#             f = open(MESH_FILE_NAME, "w")
#             f.write(fs)
#             f.close()

#         comm.Barrier()
#         mesh = dolfinx.Mesh(comm, MESH_FILE_NAME)

#         if comm.Get_rank() == 0:
#             import os

#             os.system("rm " + MESH_FILE_NAME)

#         # Apply any over-refinement specified:
#         for i in range(0, self.overRefine):
#             mesh = dolfinx.mesh.refine(mesh)

#         return mesh

#     def computeNcp(self):
#         ncp = 0
#         for s in self.splines:
#             ncp += s.getNcp()
#         return ncp

#     def getNcp(self):
#         return self.ncp

#     def getDegree(self):
#         # assumes all splines have same degree
#         return self.splines[0].getDegree()

#     def getPatchSideDofs(self, patch, direction, side, layers=1):
#         """
#         This is analogous to the ``BSpline`` method ``getSideDofs()``, but
#         it has an extra argument ``patch`` to indicate which patch to obtain
#         DoFs from.  The returned DoFs are in the global numbering.
#         """
#         localSideDofs = self.splines[patch].getSideDofs(
#             direction, side, layers)
#         retval = []
#         for dof in localSideDofs:
#             retval += [
#                 self.globalDofIndex(dof, patch),
#             ]
#         return retval


class ExplicitBSplineControlMesh(AbstractControlMesh):
    """
    A control mesh for a B-spline with identical physical and parametric
    domains.
    """

    def __init__(
        self, degrees, kvecs, extraDim=0, overRefine=0
    ):
        """
        Create an ``ExplicitBSplineControlMesh`` with degrees in each direction
        given by the sequence ``degrees`` and knot vectors given by the list
        of sequences ``kvecs``.
        """
        self.scalarSpline = BSpline(
            degrees, kvecs, overRefine=overRefine
        )
        # parametric == physical
        self.nvar = len(degrees)
        self.nsd = self.nvar + extraDim

        if self.nsd < 1 or self.nsd > 3:
            print("ERROR: Unsupported space dimension: " + str(self.nsd))
            exit()

    def getScalarSpline(self):
        return self.scalarSpline

    def getHomogeneousCoordinate(self, node, direction):
        # B-spline
        if direction == self.nsd:
            return 1.0
        # otherwise, get coordinate (homogeneous == ordniary for B-spline)
        # for explicit spline, space directions and parametric directions
        # coincide
        if direction < self.nvar:
            if self.nvar == 1:
                directionalIndex = node
            elif self.nvar == 2:
                directionalIndex = dof2ij(node, self.scalarSpline.splines[0].getNcp())[
                    direction
                ]
            else:
                M = self.scalarSpline.splines[0].getNcp()
                N = self.scalarSpline.splines[1].getNcp()
                directionalIndex = dof2ijk(node, M, N)[direction]

            # use Greville points for explicit spline
            coord = self.scalarSpline.splines[direction].greville(
                directionalIndex)
        else:
            coord = 0.0
        return coord

    def get_all_control_points(self) -> np.ndarray:
        """
        Returns all control points in homogeneous coordinates.
        """
        cp_coords = np.empty(
            (self.scalarSpline.getNcp(), self.nsd + 1), dtype=np.float64
        )

        if self.nvar == 1:
            for i in range(self.scalarSpline.getNcp()):
                cp_coords[i, 0] = self.scalarSpline.splines[0].greville(i)
                # Control points can be in 3D
                for k in range(1, self.nsd):
                    cp_coords[i, k] = 0.0
                cp_coords[i, self.nsd] = 1.0

        elif self.nvar == 2:
            # Use the tensor product structure of the B-spline
            u_spline = self.scalarSpline.splines[0]
            v_spline = self.scalarSpline.splines[1]

            n_u = u_spline.getNcp()
            n_v = v_spline.getNcp()

            u_cp_coords = np.empty(u_spline.getNcp(), dtype=np.float64)
            v_cp_coords = np.empty(v_spline.getNcp(), dtype=np.float64)

            for i in range(u_spline.getNcp()):
                u_cp_coords[i] = u_spline.greville(i)

            for j in range(v_spline.getNcp()):
                v_cp_coords[j] = v_spline.greville(j)

            for j in range(n_v):
                for i in range(n_u):
                    cp_coords[j * n_u + i, 0] = u_cp_coords[i]
                    cp_coords[j * n_u + i, 1] = v_cp_coords[j]
                    # Control points can be in 3D
                    for k in range(2, self.nsd):
                        cp_coords[j * n_u + i, k] = 0.0
                    cp_coords[j * n_u + i, self.nsd] = 1.0

        else:
            # Use the tensor product structure of the B-spline
            u_spline = self.scalarSpline.splines[0]
            v_spline = self.scalarSpline.splines[1]
            w_spline = self.scalarSpline.splines[2]

            n_u = u_spline.getNcp()
            n_v = v_spline.getNcp()
            n_w = w_spline.getNcp()

            u_cp_coords = np.empty(u_spline.getNcp(), dtype=np.float64)
            v_cp_coords = np.empty(v_spline.getNcp(), dtype=np.float64)
            w_cp_coords = np.empty(w_spline.getNcp(), dtype=np.float64)

            for i in range(u_spline.getNcp()):
                u_cp_coords[i] = u_spline.greville(i)

            for j in range(v_spline.getNcp()):
                v_cp_coords[j] = v_spline.greville(j)

            for k in range(w_spline.getNcp()):
                w_cp_coords[k] = w_spline.greville(k)

            for k in range(n_w):
                for j in range(n_v):
                    for i in range(n_u):
                        cp_coords[k * n_v * n_u + j * n_u + i, 0] = u_cp_coords[i]
                        cp_coords[k * n_v * n_u + j * n_u + i, 1] = v_cp_coords[j]
                        cp_coords[k * n_v * n_u + j * n_u + i, 2] = w_cp_coords[k]
                        cp_coords[k * n_v * n_u + j * n_u + i, 3] = 1.0

        return cp_coords

    def getNsd(self):
        return self.nsd


# # TODO: think about re-organization, as this is NURBS functionality (but
# # does not rely on igakit)
# class LegacyMultipatchControlMesh(AbstractControlMesh):
#     """
#     A class to generate a multi-patch NURBS from data given in a legacy
#     ASCII format used by some early prototype IGA codes from the Hughes
#     group at UT Austin.
#     """

#     def __init__(
#         self, prefix, nPatch, suffix, overRefine=0
#     ):
#         """
#         Loads a collection of ``nPatch`` files with names of the form
#         ``prefix+str(i+1)+suffix``, for ``i in range(0,nPatch)``, where each
#         file contains data for a NURBS patch, in the ASCII format used by
#         J. A. Cottrell's preprocessor.  (The ``+1`` in the file name
#         convention comes from Fortran indexing.)  The optional argument
#         ``overRefine`` can specify a number of global refinements of the FE
#         mesh used for extraction.  (This does not refine the IGA space.)
#         Over-refinement is only supported for simplicial elements.

#         The parametric dimension is inferred from the contents of the first
#         file, and assumed to be the same for all patches.
#         """

#         # Accummulate B-splines for each patch's scalar basis here
#         splines = []
#         # Empty control net, to be filled in with pts in homogeneous coords
#         self.bnet = []
#         # sentinel value for parametric and physical dimensions
#         nvar = -1
#         self.nsd = -1
#         for i in range(0, nPatch):

#             # Read contents of file
#             fname = prefix + str(i + 1) + suffix
#             f = open(fname, "r")
#             fs = f.read()
#             f.close()
#             lines = fs.split("\n")

#             # infer parametric dimension from the number of
#             # whitespace-delimited tokens on the second line
#             if nvar == -1:
#                 self.nsd = int(lines[0])
#                 nvar = len(lines[1].split())

#             # Load general info on $\hat{d}$, spline degrees, number of CPs
#             degrees = []
#             degStrs = lines[1].split()
#             ncps = []
#             ncpStrs = lines[2].split()
#             for d in range(0, nvar):
#                 degrees += [
#                     int(degStrs[d]),
#                 ]
#                 ncps += [
#                     int(ncpStrs[d]),
#                 ]

#             # Load knot vector for each parametric dimension
#             kvecs = []
#             for d in range(0, nvar):
#                 kvecStrs = lines[3 + d].split()
#                 kvec = []
#                 for s in kvecStrs:
#                     kvec += [
#                         float(s),
#                     ]
#                 kvecs += [
#                     np.array(kvec),
#                 ]

#             # Use the knot vectors to create a B-spline basis for this patch
#             splines += [
#                 BSpline(degrees, kvecs, overRefine),
#             ]

#             # Load control points
#             ncp = 1
#             for d in range(0, nvar):
#                 ncp *= ncps[d]

#             # Note: this only works for all parametric dimensions because
#             # the ij2dof and ijk2dof functions follow the same convention of
#             # having i as the fastest-varying index, j as the next-fastest,
#             # and k as the outer loop.
#             for pt in range(0, ncp):
#                 bnetRow = []
#                 coordStrs = lines[3 + nvar + pt].split()
#                 w = float(coordStrs[self.nsd])
#                 # NOTE: bnet should be in homogeneous coordinates
#                 for d in range(0, self.nsd):
#                     bnetRow += [
#                         float(coordStrs[d]) * w,
#                     ]
#                 bnetRow += [
#                     w,
#                 ]
#                 # Note: filling of control pts in global, multi-patch bnet is
#                 # consistent with the globalDofIndex() method of
#                 # MultiBSpline
#                 self.bnet += [
#                     bnetRow,
#                 ]

#             # TODO: formats for different parametric dimensions diverge
#             # after this point, and additional data (element types, etc.)
#             # needs to be loaded in an nvar-dependent way.  Ignoring extra
#             # data for now...

#         # create the scalar spline instance to be used for all components of
#         # the control mapping.
#         self.scalarSpline = MultiBSpline(splines)

#         # Make lookup faster
#         self.bnet = np.array(self.bnet)

#     # TODO: include some functionality to match up CPs w/in epsilon of
#     # each other and construct an IPER array.  (Cf. TODO in MultiBSpline.)
#     # Should be able to use the SciPy KD tree to do this in a few lines.

#     # Required interface for an AbstractControlMesh:
#     def getHomogeneousCoordinate(self, node, direction):
#         return self.bnet[node, direction]

#     def getScalarSpline(self):
#         return self.scalarSpline

#     def getNsd(self):
#         return self.nsd
