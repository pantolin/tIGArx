import numpy as np
import numba as nb

from tIGArx.timing_util import perf_log


def numpy_perf_test_matmul():

    for i in range(3, 12):
        N = 2**i
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)

        perf_log.start_timing("Matrix Multiplication: " + str(N) + " x " + str(N))
        C = A @ B
        perf_log.end_timing("Matrix Multiplication: " + str(N) + " x " + str(N))


def numpy_perf_test_kron():

    for i in range(2, 8):
        N = i
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        C = np.random.rand(N, N)

        perf_log.start_timing("Kronecker Product: " + str(N) + " x " + str(N))
        D = np.kron(np.kron(A, B), C)
        print(D.size * D.itemsize / 1024 / 1024)
        perf_log.end_timing("Kronecker Product: " + str(N) + " x " + str(N))


def generalized_kron_product(S1, S2, S3, I, A):
    n = S1.shape[0]
    p = I.shape[0]

    # Reshape A according to the dimensions of S1, S2, S3, and I
    # A should be reshaped to (n, n, n, p, n, n, n, p)
    A_reshaped = A.reshape(n, n, n, p, n, n, n, p)

    # Apply S3, S2, S1 to the respective dimensions
    # We avoid any operations on dimensions corresponding to I since it's identity
    temp = np.tensordot(A_reshaped, S3, axes=([2], [1]))
    temp = temp.transpose(0, 1, 4, 2, 3, 5, 6, 7)
    temp = np.tensordot(temp, S2, axes=([1], [1]))
    temp = temp.transpose(0, 4, 1, 2, 3, 5, 6, 7)
    result = np.tensordot(temp, S1, axes=([0], [1]))
    result = result.transpose(4, 0, 1, 2, 3, 5, 6, 7)

    # Reverse operations for S^T (we transpose S1, S2, S3 and handle them in reverse order)
    result = np.tensordot(result, S1.T, axes=([1], [0]))
    result = result.transpose(0, 4, 1, 2, 3, 5, 6, 7)
    result = np.tensordot(result, S2.T, axes=([2], [0]))
    result = result.transpose(0, 1, 4, 2, 3, 5, 6, 7)
    final_result = np.tensordot(result, S3.T, axes=([2], [0]))
    final_result = final_result.transpose(0, 1, 2, 5, 6, 7, 3, 4).reshape(n * n * n * p,
                                                                          n * n * n * p)

    return final_result


def generalized_kron_product_better(S1, S2, S3, I, A):
    n = S1.shape[0]
    p = I.shape[0]

    # Reshape A according to the dimensions of S1, S2, S3, and I
    # A should be reshaped to (n, n, n, p, n, n, n, p)
    A_reshaped = A.reshape(n, n, n, p, n, n, n, p)

    result = np.tensordot(
        np.tensordot(
            np.tensordot(
                A_reshaped, S3, axes=([2], [1])
            ).transpose(0, 1, 4, 2, 3, 5, 6, 7),
            S2, axes=([1], [1])
        ).transpose(0, 4, 1, 2, 3, 5, 6, 7),
        S1, axes=([0], [1])
    ).transpose(4, 0, 1, 2, 3, 5, 6, 7)

    return np.tensordot(
        np.tensordot(
            np.tensordot(
                result, S1.T, axes=([1], [0])
            ).transpose(0, 4, 1, 2, 3, 5, 6, 7),
            S2.T, axes=([2], [0])
        ).transpose(0, 1, 4, 2, 3, 5, 6, 7),
        S3.T, axes=([2], [0])
    ).transpose(0, 1, 2, 5, 6, 7, 3, 4).reshape(n * n * n * p, n * n * n * p)


# These below are commented failed prototypes
# def generalized_kron_product_einsum(S1, S2, S3, I, A):
#     n = S1.shape[0]
#     p = I.shape[0]
#
#     # Reshape A according to the dimensions of S1, S2, S3, and I
#     A_reshaped = A.reshape(n, n, n, p, n, n, n, p)
#
#     # Correcting the subscript string for the einsum operations
#     # Forward operation using einsum:
#     # Original operation involves contracting A_reshaped with S3, S2, and S1 sequentially
#     # We need to ensure we're contracting the correct dimensions
#     result = np.einsum('ijklmnop,pl,mi,nj->ijkonmop', A_reshaped, S3, S2, S1)
#
#     # Reverse operation using einsum:
#     # Now contract the result with transposes of S1, S2, S3
#     # Note: Transpose essentially flips the subscripts in the contraction
#     final_result = np.einsum('ijkonmop,oi,pj,qk->ijnmqlop', result, S1.T, S2.T, S3.T)
#
#     # Reshape the final result to match the expected output dimensions
#     final_result_reshaped = final_result.reshape(n * n * n * p, n * n * n * p)
#     return final_result_reshaped
#
#
# @nb.njit()
# def generalized_kron_product_numba(S1, S2, S3, I, A):
#     def manual_tensordot(A, B, A_axes, B_axes):
#         """
#         Manually implemented tensor dot product for 2D slices, mimicking np.tensordot(A, B, axes=([A_axes], [B_axes]))
#         Assume A and B are 4D and we are contracting over one pair of axes.
#         """
#         dim0, dim1, dim2, dim3 = A.shape
#         _, _, dim4, dim5 = B.shape
#         result = np.zeros((dim0, dim1, dim4, dim5))
#         for i in range(dim0):
#             for j in range(dim1):
#                 for k in range(dim4):
#                     for l in range(dim5):
#                         for m in range(
#                                 dim2):  # contracting over dim2 of A and dim3 of B
#                             result[i, j, k, l] += A[i, j, m, A_axes] * B[
#                                 k, l, B_axes, m]
#         return result
#
#     n = S1.shape[0]
#     p = I.shape[0]
#
#     # Reshape A according to the dimensions of S1, S2, S3, and I
#     # A should be reshaped to (n, n, n, p, n, n, n, p)
#     A_reshaped = A.reshape(n, n, n, p, n, n, n, p)
#
#     result = manual_tensordot(
#         manual_tensordot(
#             manual_tensordot(
#                 A_reshaped, S3, 2, 1
#             ).transpose(0, 1, 4, 2, 3, 5, 6, 7),
#             S2, 1, 1
#         ).transpose(0, 4, 1, 2, 3, 5, 6, 7),
#         S1, 0, 1
#     ).transpose(4, 0, 1, 2, 3, 5, 6, 7)
#
#     return manual_tensordot(
#         manual_tensordot(
#             manual_tensordot(
#                 result, S1.T, 1, 0
#             ).transpose(0, 4, 1, 2, 3, 5, 6, 7),
#             S2.T, 2, 0
#         ).transpose(0, 1, 4, 2, 3, 5, 6, 7),
#         S3.T, 2, 0
#     ).transpose(0, 1, 2, 5, 6, 7, 3, 4).reshape(n * n * n * p, n * n * n * p)


def efficient_kron_product_test():
    for i in range(2, 7):
        N = i
        d = 3
        A = np.random.rand(N**3 * d, N**3 * d)
        S1 = np.random.rand(N, N)
        S2 = np.random.rand(N, N)
        S3 = np.random.rand(N, N)
        I = np.eye(d)

        perf_log.start_timing("Efficient Product: " + str(N) + " x " + str(N))
        for _ in range (100):
            D = generalized_kron_product(S1, S2, S3, I, A)
        perf_log.end_timing("Efficient Product: " + str(N) + " x " + str(N))

        # Reference, matrix multiplication-based Kronecker product
        perf_log.start_timing("Matmul Product: " + str(N) + " x " + str(N))
        for _ in range(100):
            S = np.kron(np.kron(np.kron(S1, S2), S3), I)
            D_ref = A @ S @ A.T
        perf_log.end_timing("Matmul Product: " + str(N) + " x " + str(N))
        np.allclose(D, D_ref)


if __name__ == "__main__":
    numpy_perf_test_matmul()
    print()
    numpy_perf_test_kron()
    print()
    efficient_kron_product_test()