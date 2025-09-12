# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# System packages commonly needed (playwright will add its own deps with --with-deps too)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates git \
    fonts-liberation libasound2 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdbus-1-3 libdrm2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libxshmfence1 libgtk-3-0 libglib2.0-0 libgbm1 libnss3 libx11-xcb1 \
  && rm -rf /var/lib/apt/lists/*

# UV package manager
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy metadata first for better layer caching
COPY pyproject.toml ./
# If you later create a lock file: COPY uv.lock ./

# Install python deps into image
RUN uv sync --python 3.12

# Install Playwright browsers + system deps into image
RUN uv run playwright install --with-deps

# Copy your scraper
COPY midnattsloppet_scraper.py ./midnattsloppet_scraper.py

# Default entrypoint runs the scraper; you pass flags as usual
ENTRYPOINT ["uv", "run", "python", "midnattsloppet_scraper.py"]
CMD ["--help"]
