"""FastAPI application for Sign Language Translator."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('sign-language-translator', metadata('sign-language-translator'))
