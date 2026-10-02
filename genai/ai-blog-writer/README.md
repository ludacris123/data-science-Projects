# AI Blog Writer with RAG

Chunked RAG with default TF-IDF; `retrieval: pinecone` or `chroma` enables OpenAI dense embeddings and vector storage. Pinecone requires a provisioned 1536-dimensional cosine vector index and PINECONE_INDEX_HOST. Keys required. Token streaming remains an extension.

## Run

From repository root, use Python 3.12:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py ai-blog-writer
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

