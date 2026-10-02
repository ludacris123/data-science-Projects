"""Run from this project directory: python backend/run.py."""
import sys
from pathlib import Path
import uvicorn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

if __name__ == '__main__':
    uvicorn.run(create_app('customer-churn-saas', metadata('customer-churn-saas')), host='127.0.0.1', port=8000)
