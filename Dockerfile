FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN pip install --no-cache-dir ".[server]"

# Pre-download model
RUN python -c "from detoxify import Detoxify; Detoxify('original')"

EXPOSE 8000

CMD ["moderapi-replay", "serve", "--host", "0.0.0.0", "--port", "8000"]
