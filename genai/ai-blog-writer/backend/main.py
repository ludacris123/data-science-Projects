"""FastAPI application for AI Blog Writer with RAG."""
from portfolio_core.app import create_app
from portfolio_core.registry import metadata

app = create_app('ai-blog-writer', metadata('ai-blog-writer'))
