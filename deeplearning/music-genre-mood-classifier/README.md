# Music Genre and Mood Classifier

Librosa mel spectrograms and TensorFlow CNN. Train on NPZ arrays; labels may be genres or moods. A single classifier returns the labels it was trained on; separate genre/mood heads and playlist providers remain extensions.

## Run

From repository root, use Python 3.12:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements-tensorflow.txt
python run.py music-genre-mood-classifier
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

Train a classifier before inference:

```bash
python -m portfolio_core.train music-genre-mood-classifier --data data/music-genre-mood-classifier --epochs 10
```

For medical images: `train/<class>/`, `val/<class>/`, `test/<class>/`. For signs/audio: `train.npz`, `val.npz`, `test.npz` with `x` and integer `y`, plus `labels.json`. Sign arrays: `(samples,30,126)`; audio arrays: `(samples,128,1292,1)`. Keep patients/speakers/songs disjoint across splits. See `portfolio_core/train.py`; it exports model, label order and held-out metrics.
