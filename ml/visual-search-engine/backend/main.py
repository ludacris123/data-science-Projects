"""FastAPI application for Visual Search Engine."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('visual-search-engine', metadata('visual-search-engine'))
