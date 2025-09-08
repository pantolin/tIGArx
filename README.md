# tIGArx

A Python library for isogeometric analysis (IGA) using FEniCSx. This project is a modern translation and improvement of the original [tIGAr](https://github.com/david-kamensky/tIGAr) library from FEniCS to FEniCSx.

## Key Improvements

tIGArx introduces significant performance and memory optimizations compared to the original tIGAr:

- **Element-wise Bézier extraction**: Instead of global extraction, tIGArx uses element-wise Bézier extraction, which improves performance and reduces memory usage
- **Modern FEniCSx integration**: Full compatibility with the latest FEniCSx ecosystem

This project is maintained by Pablo Antolin with contributions from Davor Dobrota.

## Original Work

This project builds upon the foundational work by David Kamensky and Yuri Bazilevs:

```
@article{Kamensky2019,
title = "{tIGAr}: Automating isogeometric analysis with {FEniCS}",
journal = "Computer Methods in Applied Mechanics and Engineering",
volume = "344",
pages = "477--498",
year = "2019",
issn = "0045-7825",
doi = "https://doi.org/10.1016/j.cma.2018.10.002",
author = "D. Kamensky and Y. Bazilevs"
}
```

## Future Releases

A major release is expected in the coming months, focusing on:
- **Performance optimization**: Further improvements to computational efficiency
- **Simplified user interface**: Redesigned API for easier adoption
- **QUGaR interoperability**: Integration with [QUGaR](https://github.com/pantolin/qugar) for advanced quadrature methods in unfitted geometries

## Dependencies

### Core Dependencies
* [FEniCSx](https://fenicsproject.org/) (version 0.9.0) - Finite element library
* [MPICH](https://www.mpich.org/) - MPI implementation for parallel computing
* [SciPy](https://www.scipy.org/) - Scientific computing library
* [Numba](https://numba.pydata.org/) - JIT compiler for numerical functions
* [igakit](https://github.com/dalcinl/igakit) - Required for NURBS module usage
* [gfortran](https://gcc.gnu.org/fortran/) - Fortran compiler (required for building igakit)

### Optional Dependencies
* [PyVista](https://docs.pyvista.org/) - Recommended for 3D plotting and mesh analysis
* [ParaView](https://www.paraview.org/) - Recommended for visualizing results
* [Sphinx](http://www.sphinx-doc.org/en/master/) - Required for building API documentation

## Installation

### Quick Install (Recommended)

1. **Install dependencies with conda:**
   ```bash
   conda env create -f environment.yml
   conda activate tigarx
   ```

2. **Install tigarx:**
   ```bash
   pip install git+https://github.com/pantolin/tIGArx.git
   ```

### Alternative Installation Methods

**From local source:**
```bash
git clone https://github.com/pantolin/tIGArx.git
cd tIGArx
pip install -e .
```

**Development installation:**
```bash
git clone https://github.com/pantolin/tIGArx.git
cd tIGArx
pip install -e ".[dev]"
```

**Manual installation (legacy method):**
Clone the repository and add to `PYTHONPATH`:
```bash
export PYTHONPATH=/path/to/tIGArx:$PYTHONPATH
```

### Docker Installation

For a containerized environment with all dependencies pre-installed:

1. **Build the Docker image:**
   ```bash
   docker build -f docker/Dockerfile -t tigarx .
   ```

2. **Run the container:**
   ```bash
   # Interactive shell
   docker run -it tigarx
   
   # With volume mount for development
   docker run -it -v $(pwd):/app tigarx
   
   # Run a specific script
   docker run -v $(pwd):/app tigarx python demos/poisson/poisson.py
   ```

The Docker image includes:
- FEniCSx v0.9.0r1
- All required dependencies (scipy, numba, gfortran, pytest, sphinx, igakit)
- tigarx installed in development mode

### On clusters
The most convenient way to use FEniCSx likely is via [Spack](https://spack.readthedocs.io/en/latest/). See the [FEniCSx documentation](https://github.com/FEniCS/dolfinx?tab=readme-ov-file#spack) for more details.
More details on using tigarx with Spack will be provided soon.


### Documentation

To build the (_under construction_) API documentation:
```bash
cd docs
make html
```
The documentation will be available in `./_build/html/index.html`.  