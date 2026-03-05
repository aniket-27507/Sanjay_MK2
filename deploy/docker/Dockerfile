ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

LABEL maintainer="IsaacMCP" \
      description="MCP server bridging AI coding assistants with NVIDIA Isaac Sim"

WORKDIR /opt/isaac-mcp

COPY pyproject.toml README.md ./
COPY isaac_mcp/ ./isaac_mcp/
COPY config/ ./config/
COPY data/ ./data/

RUN pip install --no-cache-dir -e '.[all]' 2>/dev/null || pip install --no-cache-dir -e '.'

EXPOSE 8000

ENV ISAAC_MCP_TRANSPORT=streamable-http \
    ISAAC_MCP_HOST=0.0.0.0 \
    ISAAC_MCP_PORT=8000 \
    ISAAC_MCP_ENABLE_MUTATIONS=false

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/healthz',timeout=3); exit(0 if r.status_code==200 else 1)" || exit 1

ENTRYPOINT ["python", "-m", "isaac_mcp.server"]
CMD []
