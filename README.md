# Anime Recommendation System

[![Tests](https://github.com/tnguyen91/anime-recommendation/actions/workflows/test.yml/badge.svg)](https://github.com/tnguyen91/anime-recommendation/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/tnguyen91/anime-recommendation/branch/master/graph/badge.svg)](https://codecov.io/gh/tnguyen91/anime-recommendation)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-009688.svg)](https://fastapi.tiangolo.com)

A personalized anime recommendation engine using Restricted Boltzmann Machines (RBM). Built with FastAPI, deployed on Google Cloud Run.

**Live Demo:** [animereco-ui-725392014501.us-west1.run.app](https://animereco-ui-725392014501.us-west1.run.app)

## Features

- ML-powered recommendations using RBM neural network
- Search across 12,000+ anime
- User authentication with JWT
- Rate-limited API endpoints
- ML monitoring with drift detection
- MLflow experiment tracking

## Model Performance

| Metric | Value |
|--------|-------|
| MAP@10 | 0.4231 |
| Precision@10 | 0.1787 |
| NDCG@10 | 0.2531 |

Trained on 5,859 users × 12,347 anime (~1.3M interactions)

## Tech Stack

- **Backend:** Python 3.12, FastAPI, PyTorch, SQLAlchemy
- **Frontend:** Vanilla JS, CSS
- **Database:** PostgreSQL (Neon)
- **ML Tracking:** MLflow
- **Data Versioning:** DVC (Data Version Control)
- **Monitoring:** Streamlit dashboard, drift detection
- **Deployment:** Docker, Google Cloud Run, GitHub Actions

## Quickstart

```bash
docker-compose up --build
```

- Frontend: http://localhost:8080
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

## Project Structure

```
anime-recommendation/
├── .dvc/                   # DVC configuration
├── cache/                  # Data files (DVC tracked)
│   ├── Anime.csv.dvc       # Anime dataset pointer
│   └── User-AnimeReview.csv.dvc
├── out/                    # Model outputs (DVC tracked)
│   └── rbm_best_model.pth.dvc
├── api/                    # FastAPI backend
│   ├── auth/               # Authentication (JWT)
│   ├── favorites/          # User favorites CRUD
│   ├── inference/          # RBM model & recommendations
│   ├── monitoring.py       # Prediction logging
│   ├── alembic/            # Database migrations
│   └── tests/
├── monitoring/             # ML monitoring tools
│   ├── analyze_predictions.py
│   ├── drift_detection.py
│   └── dashboard.py        # Streamlit UI
├── data_pipeline/          # Data collection & processing
│   ├── collectors/         # Jikan API, app data export
│   ├── processors/         # Data unification, training prep
│   └── validators/         # Quality checks
├── rbm/                    # Model training
│   ├── src/                # Training code
│   ├── tests/              # Training tests
│   ├── config.yaml         # Hyperparameters
│   ├── retrain.py          # Automated retraining
│   └── hyperparameter_tuning.py
├── anime-recommendation-ui/ # Frontend
└── .github/workflows/      # CI/CD
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/recommend` | POST | Get recommendations |
| `/api/v1/search-anime` | GET | Search anime |
| `/api/v1/auth/register` | POST | Create account |
| `/api/v1/auth/login` | POST | Login |
| `/api/v1/favorites` | GET/POST/DELETE | Manage favorites |

**Example:**
```bash
curl -X POST https://animereco-api-725392014501.us-west1.run.app/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"liked_anime": ["Steins;Gate", "Death Note"]}'
```

## Development

### Local Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
uvicorn api.main:app --reload
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key variables:
- `DATABASE_URL` - PostgreSQL connection string
- `JWT_SECRET_KEY` - Secret for JWT signing (min 16 chars)
- `MODEL_URI`, `METADATA_URI`, `ANIME_CSV_URI`, `USER_REVIEW_CSV_URI` - Data/model URLs

## ML Monitoring

Predictions are logged to `logs/predictions.jsonl` with latency, inputs, and outputs.

### Analyze Predictions
```bash
python -m monitoring.analyze_predictions
```

### Detect Drift
```bash
python -m monitoring.drift_detection
```

### Dashboard
```bash
streamlit run monitoring/dashboard.py
```
Opens at http://localhost:8501

### Generate Test Data
```bash
python -m monitoring.generate_test_data
```

## Data Pipeline

Collect and process data from multiple sources:

```bash
# Full pipeline
python -m data_pipeline.run_pipeline full

# Individual steps
python -m data_pipeline.run_pipeline collect-anime
python -m data_pipeline.run_pipeline unify
python -m data_pipeline.run_pipeline validate
```

## Model Training

### Train with Best Config
```bash
python -m rbm.main --train --no-cli
```

### Hyperparameter Tuning
```bash
python -m rbm.hyperparameter_tuning
```

### View MLflow Experiments
```bash
mlflow ui
```
Opens at http://localhost:5000

### Current Best Configuration
```yaml
model:
  n_hidden: 1024
  learning_rate: 0.0005
  batch_size: 32
  epochs: 50
```

## Automated Retraining

The model can be automatically retrained with new data using the retraining pipeline.

### Manual Retraining
```bash
# Train and compare against current model
python -m rbm.retrain

# Refresh data from app database first
python -m rbm.retrain --refresh-data

# Force promote new model regardless of performance
python -m rbm.retrain --force

# Dry run (train and compare, but don't save)
python -m rbm.retrain --dry-run
```

### How It Works

1. **Data Refresh** (optional): Pulls latest user favorites from production database
2. **Training**: Trains a new RBM model with current configuration
3. **Comparison**: Evaluates new model against current production model
4. **Promotion**: Only promotes new model if it improves MAP@10 by ≥5%
5. **Logging**: All runs tracked in MLflow with full metrics

### Scheduled Retraining (GitHub Actions)

Retraining runs automatically on the 1st of each month via GitHub Actions. You can also trigger it manually:

1. Go to **Actions** → **Model Retraining**
2. Click **Run workflow**
3. Configure options:
   - `refresh_data`: Pull latest user data
   - `force_promote`: Skip performance check
   - `dry_run`: Test without saving

### Retraining Metrics

After retraining, metrics are saved to `out/retrain_metrics.json`:

```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "current_model": {"map@10": 0.42, "precision@10": 0.18},
  "new_model": {"map@10": 0.44, "precision@10": 0.19},
  "promoted": true,
  "improvement": 0.048
}
```

## Data Version Control (DVC)

Large files (datasets, models) are tracked with DVC. Git stores small pointer files (`.dvc`), while actual data lives in remote storage.

### Pull Data
```bash
pip install dvc
dvc pull
```

### After Updating Data/Model
```bash
dvc add cache/Anime.csv out/rbm_best_model.pth
dvc push
git add cache/Anime.csv.dvc out/rbm_best_model.pth.dvc
git commit -m "Update data"
```

### Tracked Files
| File | Size | Description |
|------|------|-------------|
| `cache/Anime.csv` | 12 MB | Anime metadata |
| `cache/User-AnimeReview.csv` | 100+ MB | User ratings |
| `cache/anime_metadata.json` | ~5 MB | Extended anime info |
| `out/rbm_best_model.pth` | ~50 MB | Trained model |

## Dataset

Source: [MyAnimeList Dataset](https://www.kaggle.com/datasets/bsurya27/myanimelists-anime-and-user-anime-interactions)

| Metric | Raw | Processed |
|--------|-----|-----------|
| Users | 1.2M | 5,859 |
| Anime | 28,467 | 12,347 |
| Ratings | 16.6M | 1.34M |

Preprocessing:
- Binary feedback (score ≥ 7 = liked)
- Filter users with < 50 liked anime
- Filter anime with < 50 likes
- Remove adult content

## Testing

```bash
pytest api/tests/ -v
```

## Deployment

Automated via GitHub Actions on push to `master`.

**Required GitHub Secrets:**
- `GCP_SA_KEY` - Service account JSON
- `DATABASE_URL` - PostgreSQL connection string
- `JWT_SECRET_KEY` - JWT signing key

**Required GitHub Variables:**
- `GCP_PROJECT`, `CLOUD_RUN_REGION`, `CLOUD_RUN_SERVICE`
- `MODEL_URI`, `METADATA_URI`, `ANIME_CSV_URI`, `USER_REVIEW_CSV_URI`
- `ALLOWED_ORIGINS`

## License

[MIT LICENSE](LICENSE)

Dataset attribution: [ATTRIBUTION.md](ATTRIBUTION.md)
