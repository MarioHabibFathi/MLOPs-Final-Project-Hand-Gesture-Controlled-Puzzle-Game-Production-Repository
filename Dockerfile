# Stage 1: Build environment
FROM python:3.11-slim as builder

RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libopencv-dev \
    libusb-1.0-0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install dependencies to a local directory
RUN pip install --user --no-warn-script-location -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopencv-dev \
    libusb-1.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app

# Clean Python cache
RUN find /root/.local -type d -name "__pycache__" -exec rm -rf {} + \
    && find /root/.local -type d -name "*.egg-info" -exec rm -rf {} +

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]