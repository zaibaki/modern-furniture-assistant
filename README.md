# Modern Furniture Pacific — AI Sales Assistant

Python FastAPI + LangGraph service that answers customer questions about
Modern Furniture Pacific products (Nairobi — "Make you feel luxury").
It talks to AWS Bedrock (Claude Haiku 4.5) and reads live product data
from Supabase (PostgREST, read-only).

## Endpoints

- `GET /health` — `{"status": "ok", "model": "<bedrock model id>"}`
- `POST /chat` — body `{"messages": [{"role": "user", "content": "..."}]}`,
  returns `{"answer": "...", "products": [product cards]}`. The `products`
  array holds exactly the distinct products the tools returned during
  that run, each with handle, title, category, price range, stock status,
  premium/3D image paths, product page path, WhatsApp order link and
  (for single-product lookups) variants.

## Rules the assistant follows

- Prices only from tool output, never from memory.
- Never pitches sold-out products — offers the nearest in-stock
  alternative in the same category instead.
- Delivery, discounts, negotiation and stock quantities are not in its
  data — it hands those off to WhatsApp (0716 335555), never invents them.

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SUPABASE_URL / SUPABASE_ANON_KEY
export AWS_PROFILE=redstring AWS_REGION=us-east-1
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Deploy (Cloud Run, GCP project blackbox-detective-f49a9f)

AWS keys live in Secret Manager (`mfp-assistant-aws-access-key-id`,
`mfp-assistant-aws-secret-access-key`); Supabase URL + anon key are
plain env vars (publishable by design).

```bash
gcloud run deploy modern-furniture-assistant \
  --source . --region europe-west1 --project blackbox-detective-f49a9f \
  --account thisforrichard@gmail.com --allow-unauthenticated \
  --set-env-vars SUPABASE_URL=...,SUPABASE_ANON_KEY=...,AWS_REGION=us-east-1,AWS_DEFAULT_REGION=us-east-1 \
  --set-secrets AWS_ACCESS_KEY_ID=mfp-assistant-aws-access-key-id:latest,AWS_SECRET_ACCESS_KEY=mfp-assistant-aws-secret-access-key:latest
```
