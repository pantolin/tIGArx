import numpy as np

import dolfinx

import ufl
from petsc4py import PETSc

from tigarx.LocalAssembly import assemble_matrix, assemble_vector
from tigarx.SplineInterface import AbstractScalarBasis
from tigarx.timing_util import perf_log


import platform

if platform.system() == "Windows":
    # MSVC optimization flags, effectively all CPUs have AVX2
    options = {
        "cffi_extra_compile_args": [
            "/O2", "/arch:AVX2", "/fp:fast"
        ],
    }
elif platform.system() == "Linux":
    # GCC optimization flags for Linux
    options = {
        "cffi_extra_compile_args": [
            "-O3", "-march=native", "-mtune=native", "-ffast-math"
        ],
    }
elif platform.system() == "Darwin":
    # Clang optimization flags for macOS
    if platform.machine() == "arm64":
        # ARM (Apple Silicon) with general ARM architecture, adding
        # -mcpu=apple-m1 or similar instead of -march=native is also possible
        options = {
            "cffi_extra_compile_args": [
                "-O3", "-march=native", "-ffast-math"
            ],
        }
    else:
        # Intel-based macOS - same as Linux
        options = {
            "cffi_extra_compile_args": [
                "-O3", "-march=native", "-mtune=native", "-ffast-math"
            ],
        }
else:
    # Default case - no optimization flags
    options = {
        "cffi_extra_compile_args": [],
    }


def solve_linear_variational_problem(
        lhs: ufl.form.Form,
        rhs: ufl.form.Form,
        spline: AbstractScalarBasis,
        bcs: dict[str, [np.ndarray, np.ndarray]],
        rtol=1e-12,
        profile=False,
) -> PETSc.Vec:
    """
    Solve the linear variational problem using the given forms and
    spline scalar basis. Returns the solution for control point
    coefficients in the form of a PETSc vector.

    Args:
        lhs (ufl.form.Form): left-hand side form
        rhs (ufl.form.Form): right-hand side form
        spline (AbstractScalarBasis): scalar basis
        bcs (dict[str, [np.ndarray, np.ndarray]]): boundary conditions
        profile (bool, optional): Flag to enable profiling information.
            Default is False.
        rtol (float, optional): relative tolerance for the solver.
            Default is 1e-12.

    Returns:
        PETSc.Vec: solution vector
    """
    if profile:
        perf_log.start_timing("Solving linear problem", True)
        perf_log.start_timing("Assembling problem", True)

    lhs_form = dolfinx.fem.form(lhs, jit_options=options)
    rhs_form = dolfinx.fem.form(rhs, jit_options=options)

    mat = assemble_matrix(lhs_form, spline, profile=profile)
    vec = assemble_vector(rhs_form, spline, profile=profile)

    if profile:
        perf_log.end_timing("Assembling problem")
        perf_log.start_timing("Applying boundary conditions")

    apply_bcs(mat, vec, bcs)

    if profile:
        perf_log.end_timing("Applying boundary conditions")
        perf_log.start_timing("Solving problem")

    sol = ksp_solve_iteratively(mat, vec, rtol=rtol)

    if profile:
        perf_log.end_timing("Solving problem")
        perf_log.end_timing("Solving linear problem")

    return sol


# def solve_nonlinear_variational_problem(
#         jac: ufl.form.Form,
#         res: ufl.form.Form,
#         u: dolfinx.fem.Function,
#         spline: AbstractScalarBasis,
#         extraction_func: callable,
#         bcs: dict[str, [np.ndarray, np.ndarray]],
#         rtol=1e-12,
#         profile=False,
# ) -> (PETSc.Vec, bool, int, float):
#     """
#     Solve the nonlinear variational problem using the given forms and
#     spline scalar basis. Returns the solution for control point
#     coefficients in the form of a PETSc vector.
#
#     Args:
#         jac (ufl.form.Form): Jacobian form
#         res (ufl.form.Form): residual form
#         u (dolfinx.fem.Function): initial guess and solution
#         spline (AbstractScalarBasis): scalar basis
#         extraction_func (callable): function to extract solution to FE space
#         bcs (dict[str, [np.ndarray, np.ndarray]]): boundary conditions
#         profile (bool, optional): Flag to enable profiling information.
#             Default is False.
#         rtol (float, optional): relative tolerance for the solver.
#             Default is 1e-12.
#
#     Returns:
#         PETSc.Vec: Final CP solution vector
#         bool: flag indicating convergence
#         int: number of iterations
#         float: final relative error
#     """
#     jac_form = dolfinx.fem.form(jac, jit_options=options)
#     res_form = dolfinx.fem.form(res, jit_options=options)
#
#     sol: PETSc.Vec = None
#     converged = False
#     n_iter = 0
#     ref_error = 1.0
#
#     for i in range(100):
#         if profile:
#             perf_log.start_timing("Assembling problem", True)
#
#         jac_mat = assemble_matrix(jac_form, spline, profile)
#         res_vec = assemble_vector(res_form, spline, profile)
#
#         apply_bcs(jac_mat, res_vec, bcs)
#
#         if profile:
#             perf_log.end_timing("Assembling problem")
#             perf_log.start_timing("Solving problem")
#
#         res_norm = res_vec.norm(PETSc.NormType.NORM_2)
#         if i == 0:
#             ref_error = res_norm
#         else:
#             print(f"Iteration {i} error: {res_norm / ref_error}")
#
#         rel_norm = res_norm / ref_error
#         if rel_norm < rtol:
#             converged = True
#             n_iter = i
#             ref_error = rel_norm
#             break
#
#         sol = ksp_solve_iteratively(jac_mat, res_vec, rtol=rtol)
#         extracted_sol = extraction_func(sol.array).reshape(-1)
#         u.x.array[:] -= extracted_sol
#
#         if profile:
#             perf_log.end_timing("Solving problem")
#
#     return sol, converged, n_iter, ref_error


def apply_bcs(
        mat: PETSc.Mat,
        vec: PETSc.Vec,
        bcs: dict[str, [np.ndarray, np.ndarray]],
) -> None:
    """
    Apply boundary conditions to the stiffness matrix and right-hand
    side vector

    Args:
        mat (PETSc.Mat): system matrix
        vec (PETSc.Vec): right-hand side vector
        bcs (dict[str, [np.ndarray, np.ndarray]]): boundary conditions
    """

    # TODO - improve support for different types of boundary conditions
    if bcs is not None:
        for kind, bc in bcs.items():
            bc_pos, bc_vals = bc

            if kind == "dirichlet":
                mat.zeroRowsColumns(bc_pos, 1.0)
                vec.setValues(bc_pos, bc_vals, PETSc.InsertMode.INSERT_VALUES)
            else:
                raise ValueError("Unknown boundary condition type")

    if (mat.getDiagonal().array == 0).any():
        raise RuntimeError("Cannot solve a singular system")


def ksp_solve_iteratively(A: PETSc.Mat, b: PETSc.Vec, debug=False, rtol=1e-12):
    """
    Solve the linear system Ax = b using Conjugate Gradient
    and block JACOBI preconditioning.

    Args:
        A (PETSc.Mat): The system matrix.
        b (PETSc.Vec): The right-hand side vector.
        rtol (float, optional): The relative tolerance for the solver.
            Default is 1e-12.
        profile (bool, optional): Flag to enable profiling information.
            Default is False.
    Returns:
        PETSc.Vec: The solution vector.
    """
    ksp = PETSc.KSP().create(A.getComm())
    ksp.setOperators(A)
    ksp.setType(PETSc.KSP.Type.CG)

    pc = ksp.getPC()
    pc.setType(PETSc.PC.Type.BJACOBI)

    ksp.setTolerances(rtol=rtol)

    vec = b.copy()
    if debug:
        print("-" * 60)
        print("Using CG solver with BJACOBI preconditioning")
        print(f"Matrix size:            {A.getSize()[0]}")
        info = A.getInfo()
        print(f"No. of non-zeros:       {info['nz_used']}")
        timer = dolfinx.common.Timer()
        timer.start()

    ksp.solve(b, vec)

    if debug:
        print(f"Solve took:             {timer.stop()}")
        print("-" * 60)

    vec.ghostUpdate(
        addv=PETSc.InsertMode.INSERT,
        mode=PETSc.ScatterMode.FORWARD,
    )

    return vec


def dolfinx_assemble_linear_variational_problem(
        lhs: ufl.form.Form,
        rhs: ufl.form.Form,
        profile=False,
) -> PETSc.Vec:
    """
    Test of reference dolfinx tensor assembly time
    """
    if profile:
        perf_log.start_timing("Dolfinx assembly", True)

    lhs_form = dolfinx.fem.form(lhs, jit_options=options)
    rhs_form = dolfinx.fem.form(rhs, jit_options=options)

    mat = dolfinx.fem.assemble_matrix(lhs_form)
    vec = dolfinx.fem.assemble_vector(rhs_form)

    if profile:
        perf_log.end_timing("Dolfinx assembly")

    return mat, vec
