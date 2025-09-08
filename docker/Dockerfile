FROM dolfinx/dolfinx:v0.9.0r1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    scipy \
    numba \
    pytest \
    sphinx \
    https://github.com/dalcinl/igakit/archive/refs/heads/master.zip

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install the project in development mode
RUN pip install -e .
