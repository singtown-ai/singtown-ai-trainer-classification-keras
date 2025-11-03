FROM ghcr.io/astral-sh/uv:debian

WORKDIR /app
COPY . /app

RUN cd singtown-ai-trainer-classification-keras && uv sync && uv run cache_weight.py

CMD ["sh", "run.sh"]