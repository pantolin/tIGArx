"""
The ``NURBS`` module
--------------------
leverages ``igakit`` to read in NURBS data in PetIGA's format.  The module
``igakit`` must be installed for this to be usable.
"""

from tIGAr.common import AbstractControlMesh, USE_RECT_ELEM_DEFAULT
from tIGAr.BSplines import BSpline, ij2dof, ijk2dof
from igakit.io import PetIGA
from igakit.nurbs import NURBS as NURBS_ik
import numpy as np


class NURBSControlMesh(AbstractControlMesh):
    """
    This class represents a control mesh with NURBS geometry.
    """

    def __init__(self, fname, useRect=USE_RECT_ELEM_DEFAULT, overRefine=0):
        """
        Generates a NURBS control mesh from PetIGA geometry input data
        (as generated, e.g., by a separate ``igakit`` script) in the
        file with name ``fname``.  Alternatively, ``fname`` may be an
        ``igakit`` ``NURBS`` instance, for direct construction of the
        control mesh.  The optional parameter ``useRect``
        is a Boolean specifying whether or not to use rectangular FEs for
        the extraction.
        The optional parameter ``overRefine``
        indicates how many levels of refinement to apply beyond what is
        needed to represent the spline functions; choosing a value greater
        than the default of zero may be useful for
        integrating functions with fine-scale features.
        overRefine > 0 only works for useRect=False.
        """

        # get an igakit nurbs object from the file (or, directly from an
        # existing NURBS object)
        if isinstance(fname, NURBS_ik):
            ikNURBS = fname
        else:
            ikNURBS = PetIGA().read(fname)

        # create a BSpline scalar space given the knot vector(s)
        self.scalarSpline = BSpline(ikNURBS.degree, ikNURBS.knots, useRect, overRefine)

        # get the control net; already in homogeneous form
        nvar = len(ikNURBS.degree)
        if nvar == 1:
            self.bnet = ikNURBS.control
        elif nvar == 2:
            M = ikNURBS.control.shape[0]
            N = ikNURBS.control.shape[1]
            dim = ikNURBS.control.shape[2]
            self.bnet = np.zeros((M * N, dim))
            for j in range(0, N):
                for i in range(0, M):
                    self.bnet[ij2dof(i, j, M), :] = ikNURBS.control[i, j, :]
        else:
            M = ikNURBS.control.shape[0]
            N = ikNURBS.control.shape[1]
            O = ikNURBS.control.shape[2]
            dim = ikNURBS.control.shape[3]
            self.bnet = np.zeros((M * N * O, dim))
            for k in range(0, O):
                for j in range(0, N):
                    for i in range(0, M):
                        self.bnet[ijk2dof(i, j, k, M, N), :] = ikNURBS.control[
                            i, j, k, :
                        ]

    def getScalarSpline(self):
        return self.scalarSpline

    def getHomogeneousCoordinate(self, node, direction):
        return self.bnet[node, direction]

    def getNsd(self):
        return self.bnet.shape[1] - 1
