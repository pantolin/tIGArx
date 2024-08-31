import numpy as np
import numba as nb

import dolfinx
import ufl
from petsc4py import PETSc

from tIGArx.LocalAssembly import _extract_control_points, assemble_vector, \
    assemble_matrix
from tIGArx.SplineInterface import AbstractControlMesh
from tIGArx.calculusUtils import getMetric, mappedNormal, tIGArxMeasure, volumeJacobian, \
    surfaceJacobian, pinvD, getChristoffel, cartesianGrad, cartesianDiv, cartesianCurl, \
    CurvilinearTensor, curvilinearGrad, curvilinearDiv
from tIGArx.common import selfcomm
from tIGArx.solvers import solve_linear_variational_problem, ksp_solve_iteratively, \
    apply_bcs, options
from tIGArx.timing_util import perf_log
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

    def rationalize(self, u):
        """
        Divides its argument ``u`` by the weighting function of the spline's
        control mesh.
        """
        return u / (self.control_point_funcs[-1])

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

    def project(
            self,
            ufl_expr,
            bcs: dict[str, [np.ndarray, np.ndarray]] = None,
            rationalize=True,
            lump_mass=False
    ) -> dolfinx.fem.Function:
        """
        Project a UFL expression to the finite element space.
        """
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)

        u = self.rationalize(u)
        v = self.rationalize(v)

        rhs_form = ufl.inner(ufl_expr, v) * self.dx
        ret_val = dolfinx.fem.Function(self.V)

        if not lump_mass:
            lhs_form = ufl.inner(u, v) * self.dx

            sol = solve_linear_variational_problem(
                lhs_form,
                rhs_form,
                self.spline_mesh.getScalarSpline(),
                bcs,
                profile=False
            )

            self.extract_cp_solution_to_fe(sol, ret_val)
        else:
            # TODO: Implement for dolfinx
            assert False

        if rationalize:
            ret_val = self.rationalize(ret_val)

        return ret_val

    def solve_nonlinear_variational_problem(
            self,
            jac: ufl.form.Form,
            res: ufl.form.Form,
            u: dolfinx.fem.Function,
            bcs: dict[str, [np.ndarray, np.ndarray]],
            rtol=1e-12,
            profile=False,
    ) -> (bool, int, float):
        """
        Solve the nonlinear variational problem using the given forms and
        spline scalar basis. Returns the solution for control point
        coefficients in the form of a PETSc vector.

        Args:
            jac (ufl.form.Form): Jacobian form
            res (ufl.form.Form): residual form
            u (dolfinx.fem.Function): initial guess and solution
            spline (AbstractScalarBasis): scalar basis
            bcs (dict[str, [np.ndarray, np.ndarray]]): boundary conditions
            profile (bool, optional): Flag to enable profiling information.
                Default is False.
            rtol (float, optional): relative tolerance for the solver.
                Default is 1e-12.

        Returns:
            bool: flag indicating convergence
            int: number of iterations
            float: final relative error
        """
        jac_form = dolfinx.fem.form(jac, jit_options=options)
        res_form = dolfinx.fem.form(res, jit_options=options)
        scalar_spline = self.spline_mesh.getScalarSpline()

        converged = False
        n_iter = 0
        ref_error = 1.0

        for i in range(100):
            if profile:
                perf_log.start_timing("Assembling problem", True)

            jac_mat = assemble_matrix(jac_form, scalar_spline, profile)
            res_vec = assemble_vector(res_form, scalar_spline, profile)

            apply_bcs(jac_mat, res_vec, bcs)

            if profile:
                perf_log.end_timing("Assembling problem")
                perf_log.start_timing("Solving problem")

            res_norm = res_vec.norm(PETSc.NormType.NORM_2)
            if i == 0:
                ref_error = res_norm
            else:
                if profile:
                    print(f"Iteration {i} error: {res_norm / ref_error}")

            rel_norm = res_norm / ref_error
            if rel_norm < rtol:
                converged = True
                n_iter = i
                ref_error = rel_norm
                break

            sol = ksp_solve_iteratively(jac_mat, res_vec, rtol=rtol)
            extracted_sol = self.extract_values_to_fe_cps(sol.array).reshape(-1)
            u.x.array[:] -= extracted_sol

            if profile:
                perf_log.end_timing("Solving problem")

        return converged, n_iter, ref_error