# MLX OptiQ Server with TurboQuant KV Cache
# Note: Requires Apple Silicon Mac with macOS host (Docker Desktop)
# MLX requires direct Metal GPU access - container must run on macOS

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY mlx-openai-optiq-server.py .

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HUGGINGFACE_MODEL=mlx-community/Qwen3.5-9B-OptiQ-4bit
ENV CONTEXT_LENGTH=262144

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/v1/models || exit 1

# Run server
CMD ["python", "mlx-openai-optiq-server.py"]
