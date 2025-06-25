# Anime Recommendation System with RBM and PyTorch

This project builds a GPU-accelerated anime recommendation system using a Restricted Boltzmann Machine (RBM).

---

## Project Highlights

- Collaborative filtering using RBM (unsupervised learning).
- CUDA GPU acceleration for efficient training.
- Preprocessing with filters.
- Evaluation metrics: Precision\@K, MAP\@K, NDCG\@K.
- Generates `recommendations.csv` and `training_metrics.png`.
- Recommendations for new users.

---

## Dataset

[Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) from Kaggle.

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
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Train the RBM model

```bash
python main.py --train
```

The command above downloads the Kaggle dataset (if not already), trains
the RBM model and saves the weights defined by `--model-path` (defaults to
`rbm_best_model.pth`).

If a trained model already exists you can skip training and only launch the
recommendation interface:

```bash
python main.py
```

Optional flags such as `--epochs`, `--batch-size` and `--learning-rate` control
the training process. Pass `--no-cli` to disable the interactive recommender.

### Interactive recommender

After training (or loading) a model, the script launches a CLI that lets you
search for anime you like and returns top recommendations. Type a search
keyword in anime names, select the matching `MAL_ID`s and press Enter to finish.

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
├── data/
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
