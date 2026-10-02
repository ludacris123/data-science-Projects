"""FastAPI application for E-commerce Shopping Agent."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('shopping-agent', metadata('shopping-agent'))
