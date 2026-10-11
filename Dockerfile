FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

# Extra dependency groups, e.g. "--group ml" for the fraud service.
ARG UV_SYNC_ARGS=""
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen $UV_SYNC_ARGS

COPY . .

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

# The fraud image trains its model from the fixed-seed synthetic dataset at build time,
# so no binary model is committed and a clean clone still builds a working service.
ARG TRAIN_FRAUD_MODEL=false
RUN if [ "$TRAIN_FRAUD_MODEL" = "true" ]; then \
        python -m ml.generate_data && python -m ml.train; \
    fi

CMD ["python", "-m", "uvicorn", "services.gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
