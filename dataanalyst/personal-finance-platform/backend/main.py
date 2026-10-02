"""FastAPI application for Personal Finance Analytics."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('personal-finance-platform', metadata('personal-finance-platform'))
