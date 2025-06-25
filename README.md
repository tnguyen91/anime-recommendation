# Anime Recommendation System with RBM and PyTorch

This project builds a GPU-accelerated anime recommendation system using a Restricted Boltzmann Machine (RBM).

It can train a model from the [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) and exposes a small command line interface (CLI) for getting recommendations interactively once a model is available.
---

## Project Highlights

- Collaborative filtering using RBM (unsupervised learning).
- CUDA GPU acceleration for efficient training.
- Preprocessing with filters.
- Evaluation metrics: Precision\@K, MAP\@K, NDCG\@K.
- Generates `recommendations.csv` and `training_metrics.png`.

---

## Dataset

[Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) from Kaggle, downloaded automatically using [`kagglehub`](https://github.com/Kaggle/kagglehub).

> **Note**: Accessing Kaggle datasets requires a Kaggle account and API credentials. Create a `kaggle.json` token from your Kaggle account settings and place it either in `~/.kaggle/` or set the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables before running the code.

---

## Installation

1. Clone the repo:

```bash
git clone https://github.com/yourusername/anime-recommendation.git
cd anime-recommendation
```

2. Set up the virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\\Scripts\\activate on Windows
pip install -r requirements.txt
```

## Usage

### Training

The repository does not provide a command line argument parser. To train the model edit the bottom of `main.py` so that `train_model=True`:

```python
if __name__ == "__main__":
    main(train_model=True)
```

Running the script will download the dataset, train the RBM and write metrics and recommendations to disk.

### Interactive CLI

To obtain recommendations with a trained model (e.g. `rbm_best_model.pth`), run the script with `train_model=False` (the default).

```bash
python main.py
```

### Outputs

- `recommendations.csv`: Top-10 anime per user with score and held-out flags.
- `training_metrics.png`: Training loss and evaluation metrics across epochs.
- `rbm_best_model.pth`: Best performing model weights saved during training.

---

## Evaluation Metrics

- **Precision\@K**: Fraction of top-K items that are relevant.
- **MAP\@K**: Mean Average Precision over top-K.
- **NDCG\@K**: Normalized Discounted Cumulative Gain, accounting for position.

---

## GPU Utilization

The script auto-detects and logs the available CUDA device:

```bash
Training...(using cuda)
NVIDIA GeForce RTX 4060 Laptop GPU
Memory Allocated: 286.8 MB
Memory Cached: 294.0 MB
```

---

## Project Structure

```
├── main.py
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
├── recommendations.csv
├── training_metrics.png
├── rbm_best_model.pth
```
