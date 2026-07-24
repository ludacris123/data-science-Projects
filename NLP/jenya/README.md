# LSTM Text Generation

This project trains an LSTM-based text generator on a plaintext corpus (e.g., Shakespeare / Project Gutenberg).

Usage

1. Put your dataset file in the project folder and name it `gutenberg.txt` (or pass `--data path/to/file.txt`).
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Train the model:

```
python lstm.py --train --epochs 30 --batch_size 128
```

4. Generate text (after training):

```
python lstm.py
```

Dataset

Use a public-domain text file, for example:

- Project Gutenberg — Shakespeare: https://www.gutenberg.org/ebooks/100

Notes

- The script tokenizes by word, builds incremental n-gram sequences, and trains a standard Embedding+LSTM model.
- Adjust `--epochs`, `--lstm_units`, `--embed_dim`, and `--temperature` to experiment with generation quality.
