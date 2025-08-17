# wsgi.py  (Cloud Run)
from app_cloudrun import app  # do NOT fallback
print("[WSGI] using app_cloudrun:app")

@app.get("/healthz")
def _healthz():
    return "ok", 200

