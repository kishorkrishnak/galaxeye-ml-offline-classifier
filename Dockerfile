FROM python:3.12-slim AS builder

ENV UV_PYTHON_DOWNLOADS=0
WORKDIR /app

RUN pip install --no-cache-dir uv==0.9.7
COPY pyproject.toml uv.lock ./
COPY galaxeye_ml/ ./galaxeye_ml/
RUN uv sync --locked --no-dev --no-editable --no-cache

FROM python:3.12-slim

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY --from=builder /app/.venv /app/.venv
COPY artifacts/landcover_model.joblib ./artifacts/landcover_model.joblib

EXPOSE 8000
CMD ["uvicorn", "galaxeye_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
