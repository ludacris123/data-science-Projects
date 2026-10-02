"""FastAPI application for Stock Sentiment Analyzer."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('stock-sentiment-analyzer', metadata('stock-sentiment-analyzer'))
