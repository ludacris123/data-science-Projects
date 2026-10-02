"""FastAPI application for Customer Segmentation Dashboard."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('customer-segmentation-dashboard', metadata('customer-segmentation-dashboard'))
