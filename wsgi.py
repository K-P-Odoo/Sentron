# wsgi.py
try:
    from app_cloudrun import app
    print("[WSGI] using app_cloudrun:app")
except Exception as e:
    print("[WSGI] app_cloudrun import failed, falling back to app.py:", e)
    from app import app
    print("[WSGI] using app:app")

# Ensure /healthz exists even if the loaded app doesn't define it
if not any(rule.rule == "/healthz" for rule in app.url_map.iter_rules()):
    @app.get("/healthz")
    def _healthz():
        return "ok", 200
