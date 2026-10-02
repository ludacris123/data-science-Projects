# E-commerce Shopping Agent

LangGraph search/compare and optional BrowserBase/Playwright page extraction when product_urls is supplied. Configure browser keys and allowed hosts. No Stripe checkout, purchases or scheduled price monitoring.

## Run

From repository root, use Python 3.12:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py shopping-agent
```

In a second terminal:

```bash
npm install
npm run dev --workspace @rishabh/shopping-agent
```

API documentation: http://localhost:8000/docs. The UI loads the project metadata and editable sample payload from `/schema`. POST JSON to `/run`; analytics projects also accept UTF-8 CSV uploads. All projects provide `/export/pdf` and `/export/csv`. Synthetic samples are examples, not evidence of real-world accuracy.

## Configuration and evaluation

Set environment variables from `.env.example` in your shell; secrets stay server-side. Never commit credentials. This local development server is not a production deployment. It has request-size limits and explicit CORS; it does not include authentication, tenant isolation, quotas or billing.


## Source files

- `backend/service.py`: this project's business logic.
- `backend/main.py`: FastAPI application.
- `backend/run.py`: launch from the project folder.
- `frontend/src/main.jsx`: runnable React client and project controls.
- `sample-request.json`: example API request.

From this project folder you can also run `python backend/run.py`. Shared provider, model and transport utilities remain in `portfolio_core/`.
