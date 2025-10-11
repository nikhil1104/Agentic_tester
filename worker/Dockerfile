FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY . /app

# If you keep a root requirements.txt with all deps, install that first (cache-friendly).
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
# Ensure worker deps (if not provided in root requirements)
RUN if [ -f worker/requirements.txt ]; then pip install --no-cache-dir -r worker/requirements.txt; fi

# (Optional) If you want to run Playwright UI inside this container, uncomment:
# RUN pip install playwright && playwright install --with-deps chromium

ENV WORKER_LOG_LEVEL=INFO
ENV WORKER_DEFAULT_TIMEOUT=1800
ENV RUNNER_REPORTS_DIR=/app/reports

EXPOSE 8000
CMD ["uvicorn","worker.app:app","--host","0.0.0.0","--port","8000","--workers","1"]
