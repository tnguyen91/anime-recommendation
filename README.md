# Anime Recommendation System

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
- Google Analytics integration

## Model Performance

| Metric | Value |
|--------|-------|
| MAP@10 | 0.4169 |
| Precision@10 | 0.1787 |
| NDCG@10 | 0.2531 |

Trained on 5,859 users × 12,347 anime (~1.3M interactions)

## Tech Stack

- **Backend:** Python 3.12, FastAPI, PyTorch, SQLAlchemy
- **Frontend:** Vanilla JS, CSS
- **Database:** PostgreSQL (Neon)
- **Deployment:** Docker, Google Cloud Run, GitHub Actions
- **Auth:** JWT + bcrypt

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
├── api/                    # FastAPI backend
│   ├── auth/               # Authentication (JWT)
│   ├── favorites/          # User favorites CRUD
│   ├── inference/          # RBM model & recommendations
│   ├── alembic/            # Database migrations
│   └── tests/
├── rbm/                    # Model training
│   ├── src/                # Training code
│   ├── config.yaml         # Hyperparameters
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
```env
DATABASE_URL=postgresql://...
JWT_SECRET_KEY=your-secret-key
MODEL_URI=https://github.com/tnguyen91/anime-recommendation/releases/download/v2.0/rbm_best_model.pth
METADATA_URI=https://github.com/tnguyen91/anime-recommendation/releases/download/v1.1/anime_metadata.json
ANIME_CSV_URI=https://github.com/tnguyen91/anime-recommendation/releases/download/v1.1/Anime.csv
USER_REVIEW_CSV_URI=https://github.com/tnguyen91/anime-recommendation/releases/download/v1.1/User-AnimeReview.csv
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

Supports resume - if interrupted, re-run and it will skip completed configs.

### Current Best Configuration
```yaml
model:
  n_hidden: 1024
  learning_rate: 0.001
  batch_size: 64
  epochs: 50
data:
  min_likes_user: 50
  min_likes_anime: 50
```

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

37 tests covering auth, recommendations, favorites, and validation.

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

Dataset attribution: [ATTRIBUTION.md](ATTRIBUTION.md)
