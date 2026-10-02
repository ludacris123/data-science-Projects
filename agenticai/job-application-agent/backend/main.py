"""FastAPI application for AI Job Application Agent."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('job-application-agent', metadata('job-application-agent'))
