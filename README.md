# Anime Recommendation System

Collaborative filtering recommendation system using Restricted Boltzmann Machines (RBM). Includes FastAPI backend, web UI, and Google Cloud Run deployment.

## Tech Stack

- Python 3.12, PyTorch 2.2.2, FastAPI 0.116.2
- Docker, Google Cloud Run, GitHub Actions
- Vanilla JS frontend

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
├── .github/
│   └── workflows/
│       ├── deploy-api-cloud-run.yml
│       └── deploy-ui-cloud-run.yml
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── config.py
│   ├── inference/
│   │   ├── model.py
│   │   ├── recommender.py
│   │   ├── data_loader.py
│   │   ├── preprocess.py
│   │   └── downloads.py
│   └── tests/
│       └── test_api.py
├── rbm/
│   ├── main.py
│   ├── config.yaml
│   ├── constants.py
│   ├── hyperparameter_tuning.py
│   ├── build_metadata_cache.py
│   └── src/
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── data_loader.py
│       └── utils.py
├── anime-recommendation-ui/
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── index.html
│   ├── app.js
│   ├── config.js
│   └── styles.css
├── data/
├── out/
├── docker-compose.yml
├── .env
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/recommend` | POST | Get recommendations |
| `/search-anime` | GET | Search anime |

**Request Example:**
```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"liked_anime": ["Steins;Gate", "Death Note"]}'
```

**Response:**
```json
{
  "recommendations": [
    {
      "anime_id": 9253,
      "name": "Steins;Gate",
      "title_english": "Steins;Gate",
      "title_japanese": "シュタインズ・ゲート",
      "image_url": "https://cdn.myanimelist.net/images/anime/5/73199.jpg",
      "genre": ["Sci-Fi", "Thriller"],
      "synopsis": "..."
    }
  ]
}
```

## Development Setup

### Run API
```bash
python3 -m venv venv
source venv/bin/activate
cd api && pip install -r requirements.txt
cd .. && python -m api.main
```

### Run Frontend
```bash
cd anime-recommendation-ui
python -m http.server 8080
```

### Environment Variables

Required in `.env`:
```env
MODEL_URI=https://github.com/tnguyen91/anime-recommendation/releases/download/v1.1/rbm_best_model.pth
METADATA_URI=https://github.com/tnguyen91/anime-recommendation/releases/download/v1.1/anime_metadata.json
ANIME_CSV_URI=https://github.com/tnguyen91/anime-recommendation/releases/download/v1.1/Anime.csv
USER_REVIEW_CSV_URI=https://github.com/tnguyen91/anime-recommendation/releases/download/v1.1/User-AnimeReview.csv
CACHE_DIR=/app/cache
ALLOWED_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
```

## Model Training

### Train Model
```bash
python rbm/main.py --train
```

Output: `rbm/out/rbm_best_model.pth`

### Hyperparameter Tuning
```bash
python rbm/hyperparameter_tuning.py
```

Output: `rbm/out/tuning_results.csv`

### Configuration

**API** ([api/config.py](api/config.py)):
```python
RATING_THRESHOLD = 7
DEFAULT_TOP_N = 10
MIN_LIKES_USER = 100
MIN_LIKES_ANIME = 50
N_HIDDEN = 256
```

**Training** ([rbm/config.yaml](rbm/config.yaml)):
```yaml
model:
  n_hidden: 256
  learning_rate: 0.001
  batch_size: 16
  epochs: 30
  k: 10  # Top-K for evaluation metrics

data:
  holdout_ratio: 0.1
  min_likes_user: 100
  min_likes_anime: 50
```

## Dataset

Source: [MyAnimeList Dataset](https://www.kaggle.com/datasets/bsurya27/myanimelists-anime-and-user-anime-interactions)

Preprocessing:
- Binary implicit feedback (rating ≥ 7 = liked)
- User threshold: ≥100 liked anime
- Anime threshold: ≥50 likes
- Hentai/adult content filtered

## Model

**RBM Architecture:**
- Input: Binary user-anime interaction vectors
- Hidden units: 256 (latent factors)
- Training: Contrastive Divergence (CD-1)
- Optimizer: Adam (lr=0.001)

**Evaluation Metrics:**
- Precision@10: 0.1756
- MAP@10: 0.3639
- NDCG@10: 0.1967

![Training Metrics](out/training_metrics.png)

## Testing

```bash
cd api
pytest tests/ -v
```

Coverage:
- Health checks
- Input validation
- Recommendation generation
- Metadata enrichment

## Deployment

### Google Cloud Run

Automated via GitHub Actions (`.github/workflows/deploy-api-cloud-run.yml`).

**Required Secrets:**
- `GCP_SA_KEY`

**Required Variables:**
- `GCP_PROJECT`, `CLOUD_RUN_REGION`, `CLOUD_RUN_SERVICE`
- `MODEL_URI`, `METADATA_URI`, `ANIME_CSV_URI`, `USER_REVIEW_CSV_URI`
- `ALLOWED_ORIGINS`

### Manual Docker

```bash
cd api
docker build -t anime-api .
docker run -p 8000:8000 \
  -e MODEL_URI=<url> \
  -e METADATA_URI=<url> \
  anime-api
```

## License

See [ATTRIBUTION.md](ATTRIBUTION.md) for dataset attribution.
