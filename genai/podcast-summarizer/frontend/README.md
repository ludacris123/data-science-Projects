# Podcast Summarizer and Q&A React frontend

This project owns its Vite entry point and project-specific controls. The reusable dashboard components come from the local `@rishabh/portfolio-ui` workspace package.

From the repository root:

```bash
npm install
npm run dev --workspace @rishabh/podcast-summarizer
```

Start its API with `python run.py podcast-summarizer` in another terminal. Open http://localhost:5173. A mismatched backend is detected before any workflow can run. API keys are configured on the backend only.
