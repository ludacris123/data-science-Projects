"""FastAPI application for AI Interior Design Studio."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('interior-design-studio', metadata('interior-design-studio'))
