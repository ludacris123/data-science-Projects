"""FastAPI application for Medical Image Classifier."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('medical-image-classifier', metadata('medical-image-classifier'))
