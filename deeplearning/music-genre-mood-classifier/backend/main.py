"""FastAPI application for Music Genre and Mood Classifier."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('music-genre-mood-classifier', metadata('music-genre-mood-classifier'))
