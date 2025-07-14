# Anime Recommendation System (RBM + PyTorch)

An anime recommendation system that uses a **Restricted Boltzmann Machine (RBM)** for collaborative filtering. The system provides personalized anime recommendations based on user preferences, and includes a web interface built with react and a rest api using flask.

---

## Features

- RBM-based collaborative filtering model
- CUDA GPU acceleration (auto-detected)
- Precision@K, MAP@K, and NDCG@K for evaluation
- Interactive CLI to generate top-N anime recommendations
- Grid search tuning with logged results
- Generates recommendation CSVs and training plots
- Search anime
- Get recommendations based on liked anime

---

## Dataset

**Source**: [MyAnimeList's Anime and User-Anime interactions](https://www.kaggle.com/datasets/bsurya27/myanimelists-anime-and-user-anime-interactions/data)

- Converted ratings into implicit binary format (liked = rating ≥ 7)
- Filtered out:
  - Users with < 100 liked anime
  - Anime with < 50 likes
  - Hentai genre
- Pivoted into a user-item matrix

---

## Model: Restricted Boltzmann Machine (RBM)

- Binary input vector: whether a user liked an anime
- Learns hidden representations and reconstructs unseen preferences
- Evaluated using ranking metrics over held-out anime

---

## Project Structure

```
anime-recommendation/
├── tune_hyperparameters.py
├── data_analysis.ipynb
├── api.py
├── main.py
├── config.yaml
├── requirements.txt
├── Dockerfile
├── data/
│   ├── anime_metadata.json
├── out/
│   ├── training_metrics.png
│   ├── recommendations.csv
│   └── rbm_best_model.pth
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── anime-recommender-ui/
│   ├── .env
│   ├── package.json
│   ├── public/
│   │   ├── index.html
│   │   ├── favicon.ico
│   │   └── bg.png
│   └── src/
│       ├── App.js
│       ├── App.css
│       ├── index.js
│       ├── index.css
│       ├── components/
│       │   ├── SearchBar.js
│       │   └── AnimeCard.js
│       ├── pages/
│       │   └── Home.js
│       └── utils/
│           └── api.js
```
---

## Installation

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/anime-recommendation.git
cd anime-recommendation
```

### 2. Install Python backend dependencies
```bash
pip install -r requirements.txt
```

### 3. Install frontend dependencies
```bash
cd anime-recommender-ui
npm install
```
---

## Running the App

### Start Backend + Frontend Together
```bash
cd anime-recommender-ui
npm start
```

This will:
- Start the Flask API at `http://localhost:5000`
- Launch the React UI at `http://localhost:3000`

---

## Config File

Training settings can be changed in `config.yaml`:
```yaml
model:
  n_hidden: 1024
  learning_rate: 0.001
  batch_size: 32
  epochs: 30
  k: 10
data:
  holdout_ratio: 0.1
  min_likes_user: 100
  min_likes_anime: 50
paths:
  model_path: out/rbm_best_model.pth
```

---

## Outputs

- `recommendations.csv` – top-N anime per user + hit flag
- `training_metrics.png` – training loss and evaluation metrics
- `rbm_best_model.pth` – best saved weights

---

## Evaluation Metrics

| Metric       | Description |
|--------------|-------------|
| **Precision@K** | Measures how many of the top K recommended anime were actually relevant (i.e., liked by the user). |
| **MAP@K**       | Mean Average Precision at K: captures both correctness and order of recommendations. |
| **NDCG@K**      | Normalized Discounted Cumulative Gain: gives higher weight to relevant items ranked higher. |

![Training Metrics](out/training_metrics.png)