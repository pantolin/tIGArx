# Project status and possible improvements - 11/09/2024
### Author: Davor Dobrota

## Major improvements
- Reworked the implementation of extraction from the bottom up, using the local assembly 
  approach instead of the global assembly that was used before
- Removed `ExtractionGenerator` class as an intermediate step, as it does not really
  seem necessary in this stage of the project
- Introduced new class which embodies the locally extracted spline and outfits it with
  main functionality, aligning with the previous `ExtractedSpline` class
- Added a timing utility, but it has not been propagated quite through the entire code
- The aforementioned improvements reduce the memory footprint by about p^d, where p is
  the degree of the problem and d is the dimension. There are also significant performance
  improvements in many cases (exceptions and limitations mentioned later)

### Programmers note
Most of the functionality written for this project feels a bit out of place for Python.
I would sleep a bit more easily knowing that this is all done in C/C++ with wrappers for 
high level functions. A lot of dancing around with numpy and numba was done to make it 
all work. Nonetheless, performance is usually 1.5 - 4.5 times slower than pure dolfinx 
assembly (provided that it does not run out of memory while constructing the problem).
The worse end of this spectrum is the one that is interesting in practice, where there 
are 3 or more degrees of freedom attached to each control point (3-4 times slower).
The exact reason is not entirely clear but the bottleneck is inserting the values into 
the PETSc matrix. After that it is the Kronecker product and only then MM operations.
MM operations are BLAS so one cannot do much, except use an AVX-512 server with numpy
which uses MKL, or BLIS for Zen5 workstations. OMP can help a little here, but it is 
more likely to hurt performance in general (more MPI processes is likely better).

Reworking the spline class would probably be a good idea as there is a large number of 
`if`, `elif`, and `else` cases for covering the behaviour for dimensions 1, 2, and 3 
respectively. 

The abstractions are less mature than the original code and several things that are
present there have not been copied, such as multi-patch splines. These might be useful in
practice and research. 

## Needed improvements
- **Introduce MPI**: this will perhaps be a little challenging as numpy vectors are perhaps 
  a bit too prevalent in some of the code, and one has to be careful with local indexing.
  Right now the performance is not quite at a point where I am happy with it
- Prevent OMP from going into multithreading mode: the solver and matrix assembly just 
  become slower in many 3D cases because they decide to kick in, even though they are 
  ill-suited to the tast. Simply exporting `OMP_NUM_THREADS=1` does the job but a more 
  permanent solution is welcome
- **Fix the implementation of B-spline to accommodate periodic boundary conditions**: Right
  now the B-splines just work for open-knot intervals
- Add a test where B-splines are tested with non-uniform knot vectors and perhaps with 
  varying continuities to verify that everything works in practical applications too.
  Perhaps a spinoff of Poisson
- Add tests for periodic boundary conditions (once implemented)
- **Optimize T-splines CSR pre-allocation**: the current algorithm is a brute force one which
  is designed to work with just a dofmap and without an appropriate set structure. It 
  can be considered a bottleneck in practice (about half of matrix assembly time for
  complex examples like the bouncing ball)
- **More mature abstractions**: This is mostly down to perhaps more rigorously prescribing 
  the interfaces and adding a more explicit way for mapping the parametric domain to the
  physical one. This is arguably done with `AbstractControlMesh`, but perhaps some more 
  thought is needed on the responsibilities and perhaps some specializations which would be
  more versatile than the current `LocallyExtractedSpline`
- Adding more extensive boundary conditions prescription: Right now one can specify just
  the dirichlet boundary conditions and this can only be done for the control points
- **Add application of boundary conditions on the FE domain**: thus implicitly on the
  extracted control points, and then their transformation into constraints on control points
- **Add tests with T-splines**: The bouncing ball and Reef-Knot tests are really not the best, a 
  simpler case would be a very good idea. Finding out how to make T-splines without importing
  them from a file could also be interesting, but I imagine unlikely and difficult
- Add more test cases with NURBS to make sure that everything really is correct when the 
  weights do not equal unity.
- Decide what the method `getCpDofmap` should return: Right now it either returns a numpy
  array or a numba typed list, maybe sticking with one or the other is a good idea
- Adding more accurate type annotations for `np.ndarray`
- Clear up a bit the status of compile flags for FFCx forms. THey might not be absolutely 
  essential in general, but they do provide an essentially free performance boost at the 
  expense of some compile time. 
- Figure out exactly where to use `np.int32` (PETSc matrix), and where `np.int64`: I tried
  to use `np.int32` as much as possible as it seemed natural if the matrix cannot be indexed
  with a larger value. However, this should perhaps be a dynamic in a sense
- **Enforce PETSc scalar types on all numpy arrays**
- Try to move the non-linear solver to its own separate function: This will require prescribing
  an abstraction for extracting the solution from the IGA space to FE space.
- Experiment a bit more with solvers: iterative CG with BJACOBI preconditioner seems to work
  exceptionally well overall, beating a direct solver. A comparison with pyparadisio is
  perhaps needed
- Look a little into the `numpy_perf_test.py` file to see if maybe some benefits can be had
  by not constructing the full extraction matrix. This is tentative, but it might have some
  potential, especially if a C++ implementation ever happens, or numba behaves exceptionally 
  well.
- Figure out how iterated div free solve used for Taylor-Green works
- Add test for Gauss-Legendre quadrature which was fixed from the original one which was 
  limited to go only up to order 4.
- Add lifting?