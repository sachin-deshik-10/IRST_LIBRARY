# Multi-stage build for IRST Library
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir -e ".[dev]"

# Production stage
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime AS production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash irst

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /opt/conda /opt/conda

# Copy application code
COPY --from=builder /app /app

# Change ownership to non-root user
RUN chown -R irst:irst /app

# Switch to non-root user
USER irst

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["irst-demo", "--host", "0.0.0.0", "--port", "8080"]

# Development stage
FROM builder AS development

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    ipython \
    jupyterlab \
    notebook

# Expose Jupyter port
EXPOSE 8888

# Development command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Training stage - optimized for training workloads
FROM builder AS training

# Install additional training dependencies
RUN pip install --no-cache-dir \
    wandb \
    tensorboard \
    mlflow

# Create directories for data and outputs
RUN mkdir -p /data /outputs /checkpoints

# Set volumes
VOLUME ["/data", "/outputs", "/checkpoints"]

# Training command
CMD ["irst-train", "--config", "/app/configs/default.yaml"]
