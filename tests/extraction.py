import numpy as np

from tIGArx.BSplines import compute_local_extraction_operators


def test_compute_local_extraction_operators():
    def compute_local_extraction_operators_baseline(u, p):
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
    extraction_operators = compute_local_extraction_operators(u, p)
    extraction_operators_baseline = compute_local_extraction_operators_baseline(u, p)

    print(np.allclose(extraction_operators, extraction_operators_baseline))
