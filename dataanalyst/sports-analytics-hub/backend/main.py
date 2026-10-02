"""FastAPI application for Sports Performance Hub."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('sports-analytics-hub', metadata('sports-analytics-hub'))
