"""
The ``common`` module
---------------------
contains basic definitions of abstractions for
generating extraction data and importing it again for use in analysis.  Upon
importing this module, a number of setup steps are carried out
(e.g., initializing MPI).
"""

import numpy as np
import abc
from mpi4py import MPI
from petsc4py import PETSc

from dolfinx import default_real_type
import petsc4py
import sys


petsc4py.init(sys.argv)


worldcomm = MPI.COMM_WORLD
selfcomm = MPI.COMM_SELF

mpisize = worldcomm.Get_size()
mpirank = worldcomm.Get_rank()


# FIXME to figure out right indices.
INDEX_TYPE = "int32"
# DEFAULT_PREALLOC = 100
DEFAULT_PREALLOC = 500

# Choose default behavior for permutation of indices based on the number
# of MPI tasks
if mpisize > 8:
    DEFAULT_DO_PERMUTATION = True
else:
    DEFAULT_DO_PERMUTATION = False

# TODO - most of these feel like they should be in a config file
# basis function evaluations less than this will be considered outside the
# function's support
DEFAULT_BASIS_FUNC_IGNORE_EPS = 10.0 * np.finfo(default_real_type).eps

# DEFAULT_LINSOLVER_REL_TOL = 10.0 * np.finfo(default_real_type).eps
DEFAULT_LINSOLVER_REL_TOL = 1.0e-10
DEFAULT_LINSOLVER_ABS_TOL = PETSc.DEFAULT
DEFAULT_LINSOLVER_MAX_ITERS = 1000

# This was too small for optimal convergence rates in high-order biharmonic
# discretizations with highly-refined meshes:
# DEFAULT_BASIS_FUNC_IGNORE_EPS = 1e-9


# file naming conventions
EXTRACTION_MESH_FILE = "extraction-mesh.xdmf"
EXTRACTION_INFO_FILE = "extraction-info.txt"

# FIXME is this really needed ?


def EXTRACTION_H5_CONTROL_FUNC_NAME(dim):
    return "/control" + str(dim)


EXTRACTION_ZERO_DOFS_FILE = "zero-dofs.h5"
EXTRACTION_MAT_FILE = "extraction-mat.h5"
EXTRACTION_MAT_FILE_CTRL = "extraction-mat-ctrl.h5"
EXTRACTION_VEC_FILE_CTRL_PTS = "extraction-vec-ctrl-pts.h5"

# DG space is more memory-hungry, but allows for $C^{-1}$-continuous splines,
# e.g., for div-conforming VMS, and will still work for more continuous
# spaces.
USE_DG_DEFAULT = True

# whether or not to explicitly form M^T (memory vs. speed tradeoff)
FORM_MT = False

# Helper function to generate unique temporary file names for dolfinx
# XML meshes; file name is unique for a given rank on a given communicator.
