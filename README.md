# Tenzorz / SentinelGuard IDS - FastAPI Backend

This FastAPI backend provides ML-powered endpoints for the IDS Flutter dashboard:
- `GET /health` - health check
- `GET /recent_packets` - simulated recent packets
- `POST /analyze` - upload CSV log, returns analysis summary

## Quick start (local)

1. Create virtualenv & install:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
