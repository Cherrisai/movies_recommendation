# 🎬 Movie Recommendation System

Production-level ML movie recommender — Collaborative Filtering, Content-Based, Hybrid.

## Tech Stack
| Layer | Tool |
|---|---|
| Data | Pandas · CSV (auto-generated) |
| ML | Scikit-learn · TF-IDF · Cosine Similarity |
| Notebook | Jupyter · Matplotlib · Seaborn |
| UI | Streamlit |
| API | FastAPI + Uvicorn |

## Project Structure
```
movie_recommender/
├── data/
│   ├── generate_data.py      ← generates all 4 CSVs
│   ├── movies.csv            ← 500 movies
│   ├── users.csv             ← 500 users
│   ├── ratings.csv           ← 17,000+ ratings
│   └── reviews.csv           ← 5,000 reviews
├── models/
│   ├── collab_model.pkl
│   └── content_model.pkl
├── notebooks/
│   └── movie_recommender.ipynb
├── app/
│   └── main.py               ← FastAPI
├── streamlit_app.py
├── model_utils.py
└── requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt

# Run Jupyter notebook (EDA + training)
jupyter notebook notebooks/movie_recommender.ipynb

# Launch Streamlit UI
streamlit run streamlit_app.py

# Launch FastAPI (separate terminal)
uvicorn app.main:app --reload
```

## Dataset Stats
- 500 movies across 5 industries (South Indian, Bollywood, Hollywood, Marvel, DC)
- 500 users with preference profiles
- 17,000+ ratings (zero duplicates)
- 5,000 reviews with sentiment labels

## API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| GET | /recommend/{user_id} | CF recommendations |
| GET | /similar/{movie_id} | Content-based similar |
| GET | /hybrid/{user_id} | Hybrid recs |
| GET | /movies | Browse catalog |
| GET | /movies/{movie_id} | Movie detail |
| GET | /users/{user_id} | User profile |
| GET | /reviews/{movie_id} | Movie reviews |
| GET | /health | Health check |

## Streamlit Pages
- Dashboard — dataset overview
- User Recommendations — CF personalised picks
- Similar Movies — content-based lookup
- Hybrid Mode — blended with alpha slider
- Evaluation — Precision/Recall/NDCG@K
- Browse Movies — filterable catalog
- Reviews — sentiment-filtered reviews
- API Docs — FastAPI reference
