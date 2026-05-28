"""
Shared model utilities — used by streamlit_app.py and app/main.py
"""
import os, pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

BASE   = os.path.dirname(os.path.abspath(__file__))
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")


def load_data():
    movies  = pd.read_csv(os.path.join(DATA, "movies.csv"))
    users   = pd.read_csv(os.path.join(DATA, "users.csv"))
    ratings = pd.read_csv(os.path.join(DATA, "ratings.csv"))
    reviews = pd.read_csv(os.path.join(DATA, "reviews.csv"))
    return movies, users, ratings, reviews


def build_and_save_models(movies, ratings):
    movies = movies.copy()
    movies["content_features"] = (
        movies["genres"].str.replace("|", " ", regex=False) + " " +
        movies["main_actor"] + " " + movies["director"] + " " +
        movies["industry"] + " " + movies["language"] + " " + movies["description"]
    )
    train_r, _ = train_test_split(ratings, test_size=0.2, random_state=42)
    user_enc = LabelEncoder(); prod_enc = LabelEncoder()
    tr = train_r.copy()
    tr["u"] = user_enc.fit_transform(tr["user_id"])
    tr["p"] = prod_enc.fit_transform(tr["movie_id"])
    n_u, n_p = tr["u"].nunique(), tr["p"].nunique()
    mat = csr_matrix((tr["rating"], (tr["u"], tr["p"])), shape=(n_u, n_p))
    user_sim = cosine_similarity(mat)

    tfidf    = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=5000)
    tfidf_m  = tfidf.fit_transform(movies["content_features"])
    prod_sim = cosine_similarity(tfidf_m)
    mid_to_idx = {mid: i for i, mid in enumerate(movies["movie_id"])}

    os.makedirs(MODELS, exist_ok=True)
    pickle.dump({"user_sim":user_sim,"mat":mat,"user_enc":user_enc,"prod_enc":prod_enc},
                open(os.path.join(MODELS,"collab_model.pkl"),"wb"))
    pickle.dump({"prod_sim":prod_sim,"mid_to_idx":mid_to_idx,"tfidf":tfidf},
                open(os.path.join(MODELS,"content_model.pkl"),"wb"))
    return user_sim, mat, user_enc, prod_enc, prod_sim, mid_to_idx, tfidf


def load_models(movies, ratings):
    cp = os.path.join(MODELS, "collab_model.pkl")
    tp = os.path.join(MODELS, "content_model.pkl")
    if not os.path.exists(cp) or not os.path.exists(tp):
        return build_and_save_models(movies, ratings)
    c = pickle.load(open(cp, "rb"))
    t = pickle.load(open(tp, "rb"))
    return c["user_sim"], c["mat"], c["user_enc"], c["prod_enc"], t["prod_sim"], t["mid_to_idx"], t["tfidf"]


def collab_recommend(user_id, top_n, user_sim, mat, user_enc, prod_enc, movies):
    if user_id not in user_enc.classes_:
        return pd.DataFrame()
    u_idx   = user_enc.transform([user_id])[0]
    scores  = user_sim[u_idx] @ mat.toarray()
    already = mat[u_idx].toarray().flatten()
    scores  = scores * (already == 0)
    top_i   = np.argsort(scores)[::-1][:top_n]
    mids    = prod_enc.inverse_transform(top_i)
    return movies[movies.movie_id.isin(mids)].copy()


def content_recommend(movie_id, top_n, prod_sim, mid_to_idx, movies):
    if movie_id not in mid_to_idx:
        return pd.DataFrame()
    idx    = mid_to_idx[movie_id]
    scores = sorted(enumerate(prod_sim[idx]), key=lambda x: x[1], reverse=True)
    return movies.iloc[[i for i, _ in scores[1:top_n+1]]].copy()


def hybrid_recommend(user_id, top_n, alpha, ratings,
                     user_sim, mat, user_enc, prod_enc,
                     prod_sim, mid_to_idx, movies):
    cf = collab_recommend(user_id, top_n*2, user_sim, mat, user_enc, prod_enc, movies)
    u_movies = ratings[ratings.user_id == user_id]
    if u_movies.empty or cf.empty:
        return cf.head(top_n)
    last_mid = u_movies.sort_values("movie_id").iloc[-1]["movie_id"]
    cb = content_recommend(last_mid, top_n*2, prod_sim, mid_to_idx, movies)
    cf_ids = cf["movie_id"].tolist()
    cb_ids = cb["movie_id"].tolist()
    all_ids = list(dict.fromkeys(cf_ids + cb_ids))
    cf_score = {mid: (len(cf_ids)-i)*alpha     for i,mid in enumerate(cf_ids)}
    cb_score = {mid: (len(cb_ids)-i)*(1-alpha) for i,mid in enumerate(cb_ids)}
    combined = {mid: cf_score.get(mid,0)+cb_score.get(mid,0) for mid in all_ids}
    top_ids  = sorted(combined, key=combined.get, reverse=True)[:top_n]
    return movies[movies.movie_id.isin(top_ids)].copy()
