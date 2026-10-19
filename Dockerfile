FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/triad

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt -r requirements-dev.txt

COPY . .

ENV PYTHONPATH=/opt/triad/src

RUN python -m pytest -q \
 && ruff check . \
 && mypy --strict src/triad_reconciliation

CMD ["python", "-m", "triad_reconciliation.cli.verify"]
