# syntax=docker/dockerfile:1

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# You likely don't need build tools because wheels exist for deps below.
# If you hit build issues on some platforms, uncomment the following:
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential curl ca-certificates && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source
COPY adapters /app/adapters
COPY analytics /app/analytics
COPY tools /app/tools
COPY server.py /app/server.py

# Create non-root user
RUN useradd -m appuser
USER appuser

# Default command: start the MCP server (stdio)
CMD ["python", "server.py"]
