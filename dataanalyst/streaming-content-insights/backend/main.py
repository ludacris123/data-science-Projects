"""FastAPI application for Streaming Content Insights."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('streaming-content-insights', metadata('streaming-content-insights'))
