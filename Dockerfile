FROM ghcr.io/astral-sh/uv:debian

ENV UV_NO_CACHE=true

WORKDIR /app
COPY . /app

RUN cd singtown-ai-trainer-classification-keras && uv sync && uv run cache_weight.py

CMD ["sh", "run.sh"]