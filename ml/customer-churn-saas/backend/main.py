"""FastAPI application for Customer Churn Prediction."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('customer-churn-saas', metadata('customer-churn-saas'))
