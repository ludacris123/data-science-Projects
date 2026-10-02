"""FastAPI application for Autonomous Research Assistant."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('research-assistant', metadata('research-assistant'))
