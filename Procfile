release: python -c "import playwright; playwright.install()"
web: gunicorn app:app --bind 0.0.0.0:$PORT --timeout 300 --workers 1
