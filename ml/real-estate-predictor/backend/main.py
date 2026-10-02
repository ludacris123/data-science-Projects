"""FastAPI application for Real Estate Price Predictor."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('real-estate-predictor', metadata('real-estate-predictor'))
