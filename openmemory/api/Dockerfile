FROM python:3.12-slim

LABEL org.opencontainers.image.name="mem0/openmemory-mcp"

WORKDIR /usr/src/openmemory

# Install mem0 from local fork (includes our fixes) instead of PyPI
COPY pyproject.toml README.md /tmp/mem0/
COPY mem0/ /tmp/mem0/mem0/
RUN pip install /tmp/mem0/ && rm -rf /tmp/mem0

# Install remaining dependencies (mem0ai line will be skipped since already installed)
COPY openmemory/api/requirements.txt .
RUN pip install -r requirements.txt

COPY openmemory/api/config.json .
COPY openmemory/api/ .

EXPOSE 8765
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8765"]
