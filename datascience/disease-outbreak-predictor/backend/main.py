"""FastAPI application for Disease Outbreak Predictor."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('disease-outbreak-predictor', metadata('disease-outbreak-predictor'))
