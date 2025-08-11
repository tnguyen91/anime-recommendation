# Anime Recommendation System using Restricted Boltzmann Machines

A machine learning-based anime recommendation system that leverages **Restricted Boltzmann Machines (RBM)** for collaborative filtering. The system provides personalized anime recommendations and includes a full-stack web application with React frontend and Flask API backend.

## 🚀 Features

- **Machine Learning**: RBM-based collaborative filtering model implemented with PyTorch
- **GPU Acceleration**: Automatic CUDA detection and utilization for training
- **Comprehensive Evaluation**: Precision@K, MAP@K, and NDCG@K metrics
- **Web Interface**: Modern React frontend with search and recommendation features
- **REST API**: Flask backend with comprehensive error handling and validation
- **Production Ready**: Docker containerization and Docker Compose orchestration
- **Hyperparameter Tuning**: Automated grid search with CSV result logging
- **Data Visualization**: Training metrics plots and recommendation exports

## 🛠️ Technology Stack

**Backend & ML:**
- Python 3.12
- PyTorch (RBM implementation)
- Flask (REST API)
- pandas, NumPy (data processing)
- Gunicorn (production WSGI server)

**Frontend:**
- React 18
- Modern JavaScript (ES6+)
- CSS3 with custom styling

**DevOps & Deployment:**
- Docker & Docker Compose
- Nginx (reverse proxy)
- Automated dataset downloading via KaggleHub

## 📊 Dataset & Preprocessing

**Source**: [MyAnimeList Anime and User Interactions](https://www.kaggle.com/datasets/bsurya27/myanimelists-anime-and-user-anime-interactions)

**Data Pipeline:**
- Raw ratings converted to binary implicit feedback (liked = rating ≥ 7)
- Quality filters applied:
  - Users: minimum 100 liked anime
  - Anime: minimum 50 total likes
  - Content filtering: adult/hentai genres removed
- Final format: sparse user-item interaction matrix

## 🧠 Machine Learning Model

**Restricted Boltzmann Machine (RBM):**
- **Input**: Binary user-item preference vectors
- **Architecture**: Visible units (anime) ↔ Hidden units (latent factors)
- **Training**: Contrastive Divergence with Adam optimizer
- **Evaluation**: Top-K ranking metrics on held-out interactions
- **Features**: Early stopping, model quantization, GPU acceleration

## Project Structure

```
anime-recommendation/
├── .dockerignore
├── .gitignore
├── hyperparameter_tuning.py
├── data_analysis.ipynb
├── api.py
├── main.py
├── build_metadata_cache.py
├── config.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── data/
│   ├── datasets/
│   ├── anime_metadata.json
├── out/
│   ├── tuning_results.csv
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
│   ├── Dockerfile
│   ├── .dockerignore
│   ├── package.json
│   ├── package-lock.json
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
### 2. Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Python backend dependencies
```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies
```bash
cd anime-recommender-ui
npm install
```
### 5. (Optional) Train the model
```bash
python main.py --train
```
### 6. (Optional) Hyperparameter tuning
```bash
python hyperparameters_tuning.py
```

---

## Quickstart (with Docker)

Run backend (Flask API) and frontend (React UI) together using Docker Compose:

```bash
docker-compose up --build
```

  - Frontend: http://localhost
  - API: http://localhost:5000

---

## Running Locally

### Start backend (Flask API):
```bash
python api.py
```
### Start frontend (React UI) in a new terminal:
```bash
cd anime-recommender-ui
npm start
```

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

## Environment Variables

The frontend expects a `.env` file in the `anime-recommender-ui/` directory.

1. Go to the frontend directory:
    ```bash
    cd anime-recommender-ui/
    ```

2. Copy the example environment file:
    ```bash
    cp .env.example .env
    ```

3. Edit `.env` and set the API URL as needed:

    - For Docker Compose:
      ```env
      REACT_APP_API_URL=http://backend:5000
      ```
    - For local development:
      ```env
      REACT_APP_API_URL=http://localhost:5000
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