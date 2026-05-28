"""
Movie Recommender — FastAPI
Run: uvicorn app.main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import os, sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from model_utils import load_data, load_models, collab_recommend, content_recommend, hybrid_recommend

movies, users, ratings, reviews = load_data()
user_sim, mat, user_enc, prod_enc, prod_sim, mid_to_idx, tfidf = load_models(movies, ratings)

app = FastAPI(title="Movie Recommender API", version="1.0.0",
              description="ML-powered movie recommendations — CF, Content-Based, Hybrid")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Movie(BaseModel):
    movie_id:        int
    title:           str
    industry:        str
    language:        str
    genres:          str
    main_actor:      str
    director:        str
    release_year:    int
    rating_avg:      float
    popularity_score:int
    description:     str


def df_to_movies(df: pd.DataFrame) -> List[Movie]:
    return [Movie(**{k: row[k] for k in Movie.model_fields}) for _, row in df.iterrows()]


@app.get("/health", tags=["System"])
def health():
    return {"status":"ok","movies":len(movies),"users":len(users),"ratings":len(ratings)}


@app.get("/recommend/{user_id}", tags=["Recommendations"])
def recommend(user_id:int, top_n:int=Query(default=8,ge=1,le=20)):
    """Personalised recommendations using collaborative filtering."""
    if user_id not in users["user_id"].values:
        raise HTTPException(404, f"User {user_id} not found")
    recs = collab_recommend(user_id, top_n, user_sim, mat, user_enc, prod_enc, movies)
    return {"user_id":user_id,"model":"collaborative-filtering","recommendations":df_to_movies(recs)}


@app.get("/similar/{movie_id}", tags=["Recommendations"])
def similar(movie_id:int, top_n:int=Query(default=8,ge=1,le=20)):
    """Similar movies using content-based filtering."""
    if movie_id not in movies["movie_id"].values:
        raise HTTPException(404, f"Movie {movie_id} not found")
    recs = content_recommend(movie_id, top_n, prod_sim, mid_to_idx, movies)
    return {"movie_id":movie_id,"model":"content-based","similar_movies":df_to_movies(recs)}


@app.get("/hybrid/{user_id}", tags=["Recommendations"])
def hybrid(user_id:int,
           top_n:int=Query(default=8,ge=1,le=20),
           alpha:float=Query(default=0.6,ge=0.0,le=1.0)):
    """Hybrid recommendations — CF + content-based."""
    if user_id not in users["user_id"].values:
        raise HTTPException(404, f"User {user_id} not found")
    recs = hybrid_recommend(user_id, top_n, alpha, ratings,
                            user_sim, mat, user_enc, prod_enc,
                            prod_sim, mid_to_idx, movies)
    return {"user_id":user_id,"model":f"hybrid(alpha={alpha})","recommendations":df_to_movies(recs)}


@app.get("/movies", tags=["Catalog"])
def get_movies(industry:Optional[str]=None,
               language:Optional[str]=None,
               min_rating:float=Query(default=1.0,ge=1.0,le=5.0),
               limit:int=Query(default=20,ge=1,le=100)):
    filt = movies.copy()
    if industry: filt = filt[filt["industry"].str.lower()==industry.lower()]
    if language:  filt = filt[filt["language"].str.lower()==language.lower()]
    filt = filt[filt["rating_avg"]>=min_rating].sort_values("popularity_score",ascending=False)
    return {"count":len(filt),"movies":df_to_movies(filt.head(limit))}


@app.get("/movies/{movie_id}", tags=["Catalog"])
def get_movie(movie_id:int):
    row = movies[movies.movie_id==movie_id]
    if row.empty: raise HTTPException(404,"Movie not found")
    m = row.iloc[0].to_dict()
    m["total_ratings"] = int(ratings[ratings.movie_id==movie_id].shape[0])
    m["total_reviews"] = int(reviews[reviews.movie_id==movie_id].shape[0])
    return m


@app.get("/users/{user_id}", tags=["Users"])
def get_user(user_id:int):
    row = users[users.user_id==user_id]
    if row.empty: raise HTTPException(404,"User not found")
    u = row.iloc[0].to_dict()
    u["total_ratings"]    = int(ratings[ratings.user_id==user_id].shape[0])
    u["avg_rating_given"] = round(float(ratings[ratings.user_id==user_id]["rating"].mean()),2)
    return u


@app.get("/reviews/{movie_id}", tags=["Reviews"])
def get_reviews(movie_id:int, sentiment:Optional[str]=None, limit:int=10):
    rev = reviews[reviews.movie_id==movie_id]
    if sentiment: rev = rev[rev.sentiment==sentiment.lower()]
    return {"movie_id":movie_id,"count":len(rev),"reviews":rev.head(limit).to_dict(orient="records")}
