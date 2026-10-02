# Autonomous Research Assistant

Bounded LangGraph plan → search → synthesize workflow, Tavily evidence, report citations, streamed execution steps and PDF export. OPENAI_API_KEY and TAVILY_API_KEY required. Execution events do not expose private model reasoning.

## Run

From repository root, use Python 3.12:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py research-assistant
```

In a second terminal:

```bash
cd portfolio-ui
npm install
npm run dev
```

API documentation: http://localhost:8000/docs. The UI loads the project metadata and editable sample payload from `/schema`. POST JSON to `/run`; analytics projects also accept UTF-8 CSV uploads. All projects provide `/export/pdf` and `/export/csv`. Synthetic samples are examples, not evidence of real-world accuracy.

## Configuration and evaluation

Set environment variables from `.env.example` in your shell; secrets stay server-side. Never commit credentials. This local development server is not a production deployment. It has request-size limits and explicit CORS; it does not include authentication, tenant isolation, quotas or billing.

