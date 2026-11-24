FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# No secrets in the image: SUPABASE_URL / SUPABASE_ANON_KEY come from
# Cloud Run --set-env-vars, AWS keys from Secret Manager --set-secrets.
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
