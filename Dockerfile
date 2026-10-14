FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_SYSTEM_PYTHON=1

WORKDIR /app
COPY pyproject.toml ./
COPY postpilot ./postpilot
RUN uv pip install --no-cache .

COPY config ./config
COPY knowledge ./knowledge

RUN useradd --create-home app && chown -R app /app
USER app

ENTRYPOINT ["python", "-m", "postpilot"]
CMD ["scheduler"]
