import abc

import numpy as np

from tIGArx.CoordChartSpline import AbstractCoordinateChartSpline


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

    @abc.abstractmethod
    def getControlMesh(self):
        """
        Returns some object implementing ``AbstractControlMesh``, that
        represents this spline's control mesh.
        """
        return

    @abc.abstractmethod
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
