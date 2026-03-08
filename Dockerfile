# ============================================
# Multi-stage Dockerfile for RevGate
# DNV-TC hypothesis validation pipeline
# Python 3.12 + scientific stack
# ============================================

# ============================================
# Builder stage -- install Python dependencies
# ============================================
FROM python:3.12-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Sofia

WORKDIR /build

# Build dependencies for scientific packages (numpy, scipy, networkx)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    wget \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only pyproject.toml first for layer caching
COPY pyproject.toml ./

# Install all dependencies into /root/.local
RUN pip install --user --no-cache-dir \
    pandas \
    numpy \
    pyarrow \
    scipy \
    statsmodels \
    lifelines \
    networkx \
    gseapy \
    scikit-learn \
    matplotlib \
    seaborn \
    pyyaml \
    pydantic \
    pydantic-settings \
    typer \
    requests

# Copy source and install revgate itself
COPY src/ ./src/
RUN pip install --user --no-cache-dir -e .

# ============================================
# Development stage
# ============================================
FROM python:3.12-slim AS development

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Sofia

# Runtime dependencies only
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
    ca-certificates \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application
COPY config/ ./config/
COPY src/ ./src/
COPY main.py ./

# Environment
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Cache directory -- mount as volume in production
ENV REVGATE_CACHE_DIR=/data/cache
ENV REVGATE_OUTPUT_DIR=/data/results

# Verify installation
RUN python3 main.py --help > /dev/null && \
    echo "RevGate CLI verified successfully"

# Default entrypoint -- allows: docker run revgate status
ENTRYPOINT ["python3", "main.py"]
CMD ["--help"]
