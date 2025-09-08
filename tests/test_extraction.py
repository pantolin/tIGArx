import numpy as np

from tigarx.BSplines import compute_local_bezier_extraction_operators, uniform_knots, \
    BSpline1


def test_compute_bezier_extraction_operators():
    def compute_bezier_extraction_operators_baseline(u, p):
        """
        This is a reference implementation, almost word for word,
        as implemented in the paper by Borden et al. "Isogeometric
        finite element data structure based on Bezier extraction
        of NURBS", Int. J. Numer. Meth. Engng 87, 15-47, 2011.

        Args:
            u: array of floats, Knot vector
            p: int, Polynomial degree

        Returns:
            c: numpy array of shape (n, p + 1, p + 1), Extraction operators
        """

        m = len(u)
        a = p + 1
        b = a + 1
        nb = 0
        c = [np.eye(p + 1)]

        while b < m:
            i = b

            # Count the multiplicity of the knot at location b
            while b < m and u[b] == u[b - 1]:
                b = b + 1

            mult = b - i + 1

            if mult < p:
                c.append(np.eye(p + 1))  # Initialize the next extraction operator

                # Compute the alphas
                numer = u[b - 1] - u[a - 1]
                alphas = np.zeros(p - mult)

                for j in range(p, mult, -1):
                    alphas[j - mult - 1] = numer / (u[a + j - 1] - u[a - 1])

                # Update the matrix coefficients for r new knots
                r = p - mult
                for j in range(1, r + 1):
                    s = mult + j

                    for k in range(p + 1, s, -1):
                        alpha = alphas[k - s - 1]
                        c[nb][:, k - 1] = (alpha * c[nb][:, k - 1]
                                           + (1.0 - alpha) * c[nb][:, k - 2])

                    if b < m:
                        # The range : is exclusive, so we need to add 1 to the end
                        c[nb + 1][(r - j):(r + 1), r - j] = c[nb][(p - j):(p + 1), p]

                nb += 1
                if b < m:
                    a = b
                    b = b + 1

        return np.array(c)

    u = [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]
    p = 3
    extraction_operators = compute_local_bezier_extraction_operators(np.array(u), p)
    extraction_operators_baseline = compute_bezier_extraction_operators_baseline(u, p)

    # Time the new implementation
    import time

    array = np.array(
        [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10], dtype=np.float64
    )

    start = time.time()
    for _ in range(1000):
        compute_bezier_extraction_operators_baseline(array, p)
    print("\nBaseline implementation time: ", (time.time() - start) / 1000)

    start = time.time()
    for _ in range(10000):
        compute_local_bezier_extraction_operators(array, p)
    print("New implementation time: ", (time.time() - start) / 10000)

    np.allclose(extraction_operators, extraction_operators_baseline)


def test_compute_lagrange_extraction_operators():
    knots = uniform_knots(3, 0.0, 4.0, 4)
    spline = BSpline1(3, knots)

    extraction_operators = spline.compute_local_lagrange_extraction_operator()

    import time

    start = time.time()
    knots = uniform_knots(3, 0.0, 1.0, 10)
    spline = BSpline1(3, knots)
    for _ in range(10000):
        spline.compute_local_lagrange_extraction_operator()

    print()
    print("Time: ", (time.time() - start) / 10000)

    # Reference data obtained from MATLAB by implementing word
    # for word Algorithm 1 from paper by Schillinger et al.:
    # "Lagrange extraction and projection for NURBS basis functions:
    # A direct link between isogeometric and standard nodal finite
    # element formulations", Int. J. Numer. Meth. Engng 2016
    d_1 = np.array([
        [1.000000000000000, 0.296296296296296, 0.037037037037037, 0],
        [0, 0.564814814814815, 0.518518518518519, 0.250000000000000],
        [0, 0.132716049382716, 0.395061728395062, 0.583333333333333],
        [0, 0.006172839506173, 0.049382716049383, 0.166666666666667]
    ])

    d_2 = np.array([
        [0.250000000000000, 0.074074074074074, 0.009259259259259, 0],
        [0.583333333333333, 0.549382716049383, 0.367283950617284, 0.166666666666667],
        [0.166666666666667, 0.370370370370370, 0.574074074074074, 0.666666666666667],
        [0, 0.006172839506173, 0.049382716049383, 0.166666666666667]
    ])

    d_3 = np.array([
        [0.166666666666667, 0.049382716049383, 0.006172839506173, 0],
        [0.666666666666667, 0.574074074074074, 0.370370370370370, 0.166666666666667],
        [0.166666666666667, 0.367283950617284, 0.549382716049383, 0.583333333333333],
        [0, 0.009259259259259, 0.074074074074074, 0.250000000000000]
    ])

    d_4 = np.array([
        [0.166666666666667, 0.049382716049383, 0.006172839506173, 0],
        [0.583333333333333, 0.395061728395062, 0.132716049382716, 0],
        [0.250000000000000, 0.518518518518519, 0.564814814814815, 0],
        [0, 0.037037037037037, 0.296296296296296, 1.000000000000000]
    ])

    # Stacking these arrays along a new third dimension to form a 3D tensor
    reference = np.stack((d_1, d_2, d_3, d_4), axis=0)

    np.allclose(extraction_operators, reference)


def test_compute_lagrange_extraction_operators_cont_drop():
    knots = uniform_knots(3, 0.0, 4.0, 4, continuity_drop=1)
    spline = BSpline1(3, knots)

    extraction_operators = spline.compute_local_lagrange_extraction_operator()

    # Reference data obtained from MATLAB by implementing a slightly
    # modified version of Algorithm 1 from paper by Schillinger et al.
    d_1 = np.array([
        [1.000000000000000, 0.296296296296296, 0.037037037037037, 0],
        [0, 0.444444444444445, 0.222222222222222, 0],
        [0, 0.240740740740741, 0.592592592592593, 0.500000000000000],
        [0, 0.018518518518519, 0.148148148148148, 0.500000000000000]
    ])

    d_2 = np.array([
        [0.500000000000000, 0.148148148148148, 0.018518518518519, 0],
        [0.500000000000000, 0.592592592592593, 0.240740740740741, 0],
        [0, 0.240740740740741, 0.592592592592593, 0.500000000000000],
        [0, 0.018518518518519, 0.148148148148148, 0.500000000000000]
    ])

    d_3 = np.array([
        [0.500000000000000, 0.148148148148148, 0.018518518518518, 0],
        [0.500000000000000, 0.592592592592593, 0.240740740740740, 0],
        [0, 0.240740740740741, 0.592592592592593, 0.500000000000000],
        [0, 0.018518518518519, 0.148148148148148, 0.500000000000000]
    ])

    d_4 = np.array([
        [0.500000000000000, 0.148148148148148, 0.018518518518519, 0],
        [0.500000000000000, 0.592592592592593, 0.240740740740741, 0],
        [0, 0.222222222222222, 0.444444444444444, 0],
        [0, 0.037037037037037, 0.296296296296296, 1.000000000000000]
    ])

    # Stacking these arrays along a new third dimension to form a 3D tensor
    reference = np.stack((d_1, d_2, d_3, d_4), axis=0)

    np.allclose(extraction_operators, reference)


def test_compute_lagrange_extraction_operators_elevated():
    p = 3
    knots = uniform_knots(p, 0.0, 4.0, 4, continuity_drop=0)
    spline = BSpline1(p, knots)

    extraction_operators = spline.compute_local_lagrange_extraction_operator(order=p + 1)
    print(extraction_operators)
    # Reference data obtained from MATLAB by implementing a slightly
    # modified version of Algorithm 1 from paper by Schillinger et al.
    d_1 = np.array([
        [1.000000000000000, 0.421875000000000, 0.125000000000000, 0.015625000000000, 0],
        [0, 0.496093750000000, 0.593750000000000, 0.457031250000000, 0.250000000000000],
        [0, 0.079427083333333, 0.260416666666667, 0.457031250000000, 0.583333333333333],
        [0, 0.002604166666667, 0.020833333333333, 0.070312500000000, 0.166666666666667]
    ])

    d_2 = np.array([
        [0.250000000000000, 0.105468750000000, 0.031250000000000, 0.003906250000000, 0],
        [0.583333333333333, 0.576822916666667, 0.468750000000000, 0.313802083333333,
         0.166666666666667],
        [0.166666666666667, 0.315104166666667, 0.479166666666667, 0.611979166666667,
         0.666666666666667],
        [0, 0.002604166666667, 0.020833333333333, 0.070312500000000, 0.166666666666667]
    ])

    d_3 = np.array([
        [0.166666666666667, 0.070312500000000, 0.020833333333333, 0.002604166666667, 0],
        [0.666666666666667, 0.611979166666667, 0.479166666666667, 0.315104166666667,
         0.166666666666667],
        [0.166666666666667, 0.313802083333333, 0.468750000000000, 0.576822916666667,
         0.583333333333333],
        [0, 0.003906250000000, 0.031250000000000, 0.105468750000000, 0.250000000000000]
    ])

    d_4 = np.array([
        [0.166666666666667, 0.070312500000000, 0.020833333333333, 0.002604166666667, 0],
        [0.583333333333333, 0.457031250000000, 0.260416666666667, 0.079427083333333, 0],
        [0.250000000000000, 0.457031250000000, 0.593750000000000, 0.496093750000000, 0],
        [0, 0.015625000000000, 0.125000000000000, 0.421875000000000, 1.000000000000000]
    ])

    # Stacking these arrays along a new third dimension to form a 3D tensor
    reference = np.stack((d_1, d_2, d_3, d_4), axis=0)

    np.allclose(extraction_operators, reference)
