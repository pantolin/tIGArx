import numpy as np
import numba as nb

import dolfinx
import ufl
from petsc4py import PETSc

from tIGArx.LocalAssembly import get_full_operator
from tIGArx.SplineInterface import AbstractControlMesh, AbstractScalarBasis
from tIGArx.calculusUtils import getMetric, mappedNormal, tIGArxMeasure, volumeJacobian, \
    surfaceJacobian, pinvD, getChristoffel, cartesianGrad, cartesianDiv, cartesianCurl, \
    CurvilinearTensor, curvilinearGrad, curvilinearDiv
from tIGArx.common import selfcomm
from tIGArx.utils import createFunctionSpace, get_lagrange_permutation, \
    createElementType, createVectorElementType


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

        self.control_element = createElementType(
            self.spline_mesh.getScalarSpline().getDegree(),
            self.space_dim,
            discontinuous=False
        )
        self.control_space = createFunctionSpace(self.mesh, self.control_element)

        self.control_point_funcs = [
            dolfinx.fem.Function(self.control_space) for _ in range(self.space_dim + 1)
        ]

        self.space_element = createVectorElementType(
            [self.spline_mesh.getScalarSpline().getDegree()] * self.dofs_per_cp,
            self.space_dim,
            discontinuous=False,
            nFields=self.dofs_per_cp
        )
        self.V = createFunctionSpace(self.mesh, self.space_element)

        self.control_points: np.ndarray | None = None
        self.extracted_control_points: np.ndarray | None = None
        self.boundary_markers = None

    @staticmethod
    def get_from_mesh_and_init(
            mesh: AbstractControlMesh,
            comm=selfcomm,
            quad_degree=2,
            dofs_per_cp=1
    ):
        spline = LocallyConstructedSpline(mesh, comm, quad_degree, dofs_per_cp)

        spline.init_control_points()
        spline.init_extracted_control_points()
        spline.init_ufl_symbols()

        return spline

    def init_control_points(self):
        """
        Obtain the control points from the spline mesh.
        """
        self.control_points = self.spline_mesh.get_all_control_points()

    def init_extracted_control_points(self):
        """
        Initialize the control points of the spline by extracting the values
        from the control point functions.
        """

        self.extracted_control_points = (
            self.extract_values_to_fe_cps(self.control_points)
        )

        size = self.control_point_funcs[0].x.index_map.size_local

        for i in range(self.space_dim):
            self.control_point_funcs[i].x.array[:size] = (
                self.extracted_control_points[:, i] / self.extracted_control_points[:, -1]
            )

        self.control_point_funcs[-1].x.array[:size] = np.ones(size)

    def init_ufl_symbols(self):
        tag = 0
        face_dim = self.mesh.topology.dim - 1
        all_facets = np.arange(*self.mesh.topology.index_map(face_dim).local_range)

        self.boundary_markers = dolfinx.mesh.meshtags(
            self.mesh,
            self.mesh.topology.dim - 1,
            all_facets,
            np.full_like(all_facets, tag, dtype=np.int32))

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
                self.g, self.N), ufl.ds, self.quad_degree, self.boundary_markers
        )

        # useful for defining Cartesian differential operators
        self.pinvDF = pinvD(self.F)

        # useful for tensors given in parametric coordinates
        self.gamma = getChristoffel(self.g)

    def extract_values_to_fe_cps(self, values: np.ndarray) -> np.ndarray:
        """
        Extract the values to the control points of the finite element space.
        """
        # The number of values must match the number of control points.
        assert values.shape[0] == self.control_points.shape[0]

        if len(values.shape) == 1:
            values = values[:, np.newaxis]

        values = np.concatenate(
            (values, np.ones((values.shape[0], 1), dtype=np.float64)),
            axis=1
        )

        scalar_spline = self.spline_mesh.getScalarSpline()
        cells = np.arange(self.mesh.topology.original_cell_index.size, dtype=np.int32)
        fe_dofmap = self.control_space.dofmap.list

        extraction_dofmap = scalar_spline.getExtractionOrdering(self.mesh)
        spline_dofmap = scalar_spline.getCpDofmap(extraction_dofmap)
        extraction_operators = scalar_spline.get_lagrange_extraction_operators()

        extracted_values = np.zeros(
            (self.control_space.dofmap.index_map.size_local, values.shape[1]),
            dtype=np.float64
        )

        perm = get_lagrange_permutation(
            self.control_space.element.basix_element.points,
            self.spline_mesh.getScalarSpline().getDegree(),
            self.space_dim
        )

        _extract_control_points(
            cells,
            spline_dofmap,
            extraction_dofmap,
            extraction_operators,
            fe_dofmap,
            perm,
            values,
            extracted_values,
            self.space_dim
        )

        extracted_values[:, :-1] /= extracted_values[:, -1][:, np.newaxis]

        return extracted_values[:, :-1]

    def extract_cp_solution_to_fe(self, cp_sol: PETSc.Vec, fe_sol: dolfinx.fem.Function):
        """
        Extract the solution at the control points to the finite element space.
        """
        sol = self.extract_values_to_fe_cps(
            cp_sol.array_r.reshape(-1, self.dofs_per_cp)
        )
        size = fe_sol.x.index_map.size_local * fe_sol.x.block_size
        fe_sol.x.array[:size] = sol.reshape(-1)

    def get_fe_cp_coordinates(self):
        """
        Return the coordinates of the control points in the finite element space.
        """
        return self.F

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


@nb.jit(nopython=True, nogil=True, cache=True, fastmath=True)
def _extract_control_points(
        cells,
        spline_dofmap,
        extraction_dofmap,
        extraction_operators,
        fe_dofmap,
        permutation,
        control_points,
        extracted_control_points,
        space_dim,
):
    for cell in cells:
        element = extraction_dofmap[cell]
        full_operator = get_full_operator(extraction_operators, 1, space_dim, element)

        local_cp_range = spline_dofmap[cell]
        local_fe_range = fe_dofmap[cell][permutation]

        extracted_control_points[local_fe_range, :] += (
            full_operator.T @ (control_points[local_cp_range, :])
        )
