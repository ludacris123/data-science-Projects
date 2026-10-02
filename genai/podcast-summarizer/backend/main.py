"""FastAPI application for Podcast Summarizer and Q&A."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('podcast-summarizer', metadata('podcast-summarizer'))
