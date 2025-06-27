# Anime Recommendation System with RBM and PyTorch

This project provides a GPU-accelerated anime recommendation system using a Restricted Boltzmann Machine (RBM), implemented in PyTorch. It trains on user-anime ratings data from Kaggle and generates personalized anime recommendations.

---

## Project Highlights

- Collaborative filtering using RBM (unsupervised learning)
- CUDA GPU acceleration for efficient training
- Flexible preprocessing and filtering of input data
- Evaluation metrics: Precision@K, MAP@K, NDCG@K
- Interactive CLI for generating recommendations for new users
- Hyperparameter tuning support via `tune_hyperparameters.py`
- Customizable via `config.yaml`

---

## Dataset

- [Anime Recommendation Database 2020](https://www.kaggle.com/datasets/hernan4444/anime-recommendation-database-2020) from Kaggle.
- Data will be downloaded automatically if not present.

---

## Installation

1. Clone the repo:

    ```bash
    git clone https://github.com/tnguyen91/anime-recommendation.git
    cd anime-recommendation
    ```

2. Set up the virtual environment and install dependencies:

    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

---

## Usage

### Train the RBM model

```bash
python main.py --train
```

- Downloads the Kaggle dataset (if not already present), trains the RBM model, and saves the weights (default: `rbm_best_model.pth`).
- Optional flags:
    - `--epochs`, `--batch-size`, `--learning-rate` for customizing training.
    - `--model-path` to specify model checkpoint file.
    - `--no-cli` to disable the interactive recommendation interface.

### Launch Interactive Recommender

If a trained model already exists, you can skip training and launch the recommendation interface:

```bash
python main.py
```

- The CLI lets you search for anime you like, select by `MAL_ID`, and receive top recommendations.

### Hyperparameter Tuning

To tune hyperparameters automatically, use:

```bash
python tune_hyperparameters.py
```

- Results are saved to `tuning_results.csv`.

### Outputs

- `recommendations.csv`: Top-10 anime per user with scores and held-out flags.
- `training_metrics.png`: Training loss and evaluation metrics across epochs.
- `rbm_best_model.pth`: Best performing model weights saved during training.
- `tuning_results.csv`: Hyperparameter tuning logs and best parameters.

---

## Configuration

- Modify `config.yaml` to change training, model, or data processing settings.

---

## Evaluation Metrics

- **Precision@K**: Fraction of top-K items that are relevant.
- **MAP@K**: Mean Average Precision over top-K.
- **NDCG@K**: Normalized Discounted Cumulative Gain, accounting for ranking position.

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
├── tune_hyperparameters.py  
├── config.yaml              
├── requirements.txt         
├── data/
│   └── datasets/            
├── src/
│   ├── data_loader.py       
│   ├── model.py             
│   ├── train.py             
│   ├── evaluate.py          
│   └── utils.py             
├── recommendations.csv      
├── training_metrics.png     
├── rbm_best_model.pth       
├── tuning_results.csv       
└── data_analysis.ipynb      
```
