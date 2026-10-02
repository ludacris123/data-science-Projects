# AI and analytics portfolio — Rishabh Patel

18 local full-stack prototypes grouped into `datascience/`, `dataanalyst/`, `ml/`, `deeplearning/`, `genai/`, and `agenticai/`. Each project has its own `backend/service.py` business logic and runnable `frontend/src/main.jsx` React client. Reusable FastAPI transport and provider utilities live in `portfolio_core/`; shared React components live in `portfolio-ui/`. Neural model builders and training launchers are in the deep-learning project folders.

See [all projects](PROJECTS.md). Existing repository projects are preserved.

## What is ready

Eight tabular analytical workflows run with included synthetic samples. Four TensorFlow services implement image similarity or inference using trained artifacts. Three GenAI and three agent services integrate real external providers and require service keys. Every project has an API and shares the React dashboard. This is a portfolio prototype collection, not 18 production services. Dataset ingestion, third-party platforms and advanced requested features are identified individually in the project READMEs.

## Validation

```bash
pip install -r requirements.txt
python -m pytest tests -q
npm ci
npm run build
```

TensorFlow inference, training, live provider calls and clinical/financial accuracy require separate evaluation. Synthetic examples must not be presented as benchmark performance.

## Design

Keep analytics logic separate from HTTP transport. Uploads are bounded. CSV exports escape spreadsheet formula prefixes. No API keys go to the browser. Hosted AI providers are called at fixed domains with timeouts. Research agents have three searches per run, expose execution events, and do not perform purchases or submit job applications. Model outputs and citations still require user review.

## TensorFlow decisions

TensorFlow/Keras replaces PyTorch in the code we train and serve. Visual similarity supports EfficientNet and KerasHub CLIP with a TensorFlow backend. Hosted Whisper and diffusion APIs execute at their provider and do not require a local PyTorch package.

References: https://keras.io/api/applications/efficientnet/efficientnet_models/ and https://reference.langchain.com/python/langgraph/graph/state/StateGraph

## Run a project-specific frontend

```bash
npm install
npm run dev --workspace @rishabh/ai-blog-writer
```

Start its matching API with `python run.py ai-blog-writer`. Repeat with any project ID in PROJECTS.md. The frontend checks that the connected API matches its project.
