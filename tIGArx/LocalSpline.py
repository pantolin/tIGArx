import numpy as np
import numba as nb

import dolfinx
import ufl
from numpy.random import permutation

from tIGArx.LocalAssembly import get_full_operator
from tIGArx.SplineInterface import AbstractControlMesh
from tIGArx.calculusUtils import getMetric, mappedNormal, tIGArxMeasure, volumeJacobian, \
    surfaceJacobian, pinvD, getChristoffel
from tIGArx.common import selfcomm
from tIGArx.utils import create_ufl_vector_element, createFunctionSpace, \
    get_lagrange_permutation, createElementType


class LocallyConstructedSpline:
    def __init__(
            self,
            mesh: AbstractControlMesh,
            comm=selfcomm,
            quad_degree=2,
            dofs_per_cp=1
    ):
        self.spline_mesh: AbstractControlMesh = mesh
        self.comm = comm
        self.quad_degree = quad_degree
        self.dofs_per_cp = dofs_per_cp

        self.mesh: dolfinx.mesh.Mesh = mesh.getScalarSpline().generateMesh()
        self.space_dim = mesh.getNsd()

        self.element_control = createElementType(
            self.spline_mesh.getScalarSpline().getDegree(),
            self.space_dim,
            discontinuous=False
        )
        self.control_space = createFunctionSpace(self.mesh, self.element_control)

        self.control_point_funcs = [
            dolfinx.fem.Function(self.control_space) for _ in range(self.space_dim + 1)
        ]

        self.control_points = None
        self.extracted_control_points = None

    def init_extracted_control_points(self):
        """
        Initialize the control points of the spline by extracting the values
        from the control point functions.
        """
        self.control_points = self.spline_mesh.get_all_control_points()

        scalar_spline = self.spline_mesh.getScalarSpline()
        # cells = scalar_spline.getExtractionOrdering(self.mesh)
        cells = np.arange(self.mesh.topology.original_cell_index.size, dtype=np.int32)

        spline_dofmap = scalar_spline.getCpDofmap(cells)
        extraction_operators = scalar_spline.get_lagrange_extraction_operators()
        # fe_dofmap = self.control_space.dofmap.list
        fe_dofmap = scalar_spline.getFEDofmap(cells)

        self.extracted_control_points = np.zeros(
            (self.control_space.dofmap.index_map.size_local, self.space_dim + 1),
            dtype=np.float64
        )

        _extract_control_points(
            cells,
            spline_dofmap,
            extraction_operators,
            fe_dofmap,
            self.control_points,
            self.extracted_control_points,
            self.space_dim
        )

        permutation = self.control_space.tabulate_dof_coordinates()

        size = self.control_point_funcs[0].x.index_map.size_local

        for i in range(self.space_dim):
            self.control_point_funcs[i].x.array[:size] = (
                self.extracted_control_points[:, i] / self.extracted_control_points[:, -1]
            )

        self.control_point_funcs[-1].x.array[:size] = np.ones(size)

    def init_ufl_symbols(self):
        components = []
        for i in range(0, self.space_dim):
            components += [
                self.control_point_funcs[i] / self.control_point_funcs[self.space_dim]
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
        self.dx = tIGArxMeasure(volumeJacobian(self.g), ufl.dx, self.quad_degree)
        self.ds = tIGArxMeasure(
            surfaceJacobian(
                self.g, self.N), ufl.ds, self.quad_degree, self.boundaryMarkers
        )

        # useful for defining Cartesian differential operators
        self.pinvDF = pinvD(self.F)

        # useful for tensors given in parametric coordinates
        self.gamma = getChristoffel(self.g)


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _extract_control_points(
        cells,
        spline_dofmap,
        extraction_operators,
        fe_dofmap,
        control_points,
        extracted_control_points,
        space_dim,
):
    for cell in cells:
        full_operator = get_full_operator(extraction_operators, 1, space_dim, cell)

        local_cp_range = spline_dofmap[cell]
        local_fe_range = fe_dofmap[cell]

        extracted_control_points[local_fe_range, :] += (
            full_operator.T @ (control_points[local_cp_range, :])
        )
