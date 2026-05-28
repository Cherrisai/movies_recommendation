"""
Movie Recommendation System — Streamlit UI
Run: streamlit run app.py
"""
import os, sys
import streamlit as st
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from model_utils import load_data, load_models, collab_recommend, content_recommend, hybrid_recommend

st.set_page_config(page_title="Movie Recommender Dashboard", page_icon="", layout="wide")

st.markdown("""
<style>
.movie-card {background:#1e293b;border-radius:12px;padding:14px 18px;
             border:1px solid #334155;margin-bottom:10px;color:#e2e8f0;}
.movie-title{font-size:15px;font-weight:700;color:#38bdf8;margin:0 0 5px;}
.movie-meta {font-size:12px;color:#94a3b8;margin:0;}
.badge      {padding:3px 9px;border-radius:99px;font-size:11px;font-weight:600;margin-right:4px;}
.b-industry {background:#1e40af;color:#bfdbfe;}
.b-genre    {background:#14532d;color:#bbf7d0;}
.b-rating   {background:#78350f;color:#fde68a;}
.kpi-card   {background:#0f172a;border-radius:12px;padding:18px;text-align:center;border:1px solid #1e293b;}
.kpi-val    {font-size:28px;font-weight:700;color:#38bdf8;margin:0;}
.kpi-lbl    {font-size:13px;color:#64748b;margin:0;}
.section-hdr{background:linear-gradient(90deg,#0f172a,#1e3a5f);color:white;
             padding:10px 18px;border-radius:10px;font-size:18px;font-weight:700;margin-bottom:15px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models...")
def get_everything():
    movies, users, ratings, reviews = load_data()
    user_sim, mat, user_enc, prod_enc, prod_sim, mid_to_idx, tfidf = load_models(movies, ratings)
    return movies, users, ratings, reviews, user_sim, mat, user_enc, prod_enc, prod_sim, mid_to_idx, tfidf

movies, users, ratings, reviews, user_sim, mat, user_enc, prod_enc, prod_sim, mid_to_idx, tfidf = get_everything()


def render_movie_card(row, rank=None):
    rank_s = f'<span style="color:#64748b;font-size:12px">#{rank}</span> ' if rank else ""
    stars  = "★" * int(round(float(row.get("rating_avg", 0))))
    genres = " ".join([f'<span class="badge b-genre">{g}</span>' for g in str(row["genres"]).split("|")[:2]])
    st.markdown(f"""
    <div class="movie-card">
      <p class="movie-title">{rank_s}{row['title']}</p>
      <p class="movie-meta">
        <span class="badge b-industry">{row['industry']}</span>
        {genres}
        <span class="badge b-rating">{row.get('rating_avg','-')} {stars}</span>
      </p>
      <p class="movie-meta" style="margin-top:4px;">
        🎭 {row['main_actor']} &nbsp;|&nbsp; 🎬 {row['director']} &nbsp;|&nbsp; {row['release_year']}
      </p>
      <p class="movie-meta" style="margin-top:4px;font-style:italic;">{str(row['description'])[:120]}...</p>
    </div>
    """, unsafe_allow_html=True)


def kpi(col, val, label):
    col.markdown(f"""<div class="kpi-card">
        <p class="kpi-val">{val}</p>
        <p class="kpi-lbl">{label}</p></div>""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Movie Recommender")
    st.caption("ML-powered personalised movie suggestions")
    st.markdown("---")
    page = st.radio("", [
        "Dashboard",
        "User Recommendations",
        "Similar Movies",
        "Hybrid Mode",
        "Evaluation",
        "Browse Movies",
        "Reviews",
        "API Docs",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption(f"**{len(movies)}** movies · **{len(users)}** users")
    st.caption(f"**{len(ratings):,}** ratings · **{len(reviews):,}** reviews")

# ── Dashboard ─────────────────────────────────────────────────────────────────
if page == "Dashboard":
    st.markdown('<div class="section-hdr">Dataset Overview</div>', unsafe_allow_html=True)
    c1,c2,c3,c4,c5 = st.columns(5)
    kpi(c1, len(movies),       "Movies")
    kpi(c2, len(users),        "Users")
    kpi(c3, f"{len(ratings):,}","Ratings")
    kpi(c4, f"{len(reviews):,}","Reviews")
    kpi(c5, movies['industry'].nunique(), "Industries")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Movies by Industry")
        ic = movies["industry"].value_counts().reset_index()
        ic.columns = ["Industry","Count"]
        st.bar_chart(ic.set_index("Industry"))
    with col2:
        st.markdown("#### Rating Distribution")
        rc = ratings["rating"].value_counts().sort_index().reset_index()
        rc.columns = ["Rating","Count"]
        st.bar_chart(rc.set_index("Rating"))

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Sentiment Distribution")
        sc = reviews["sentiment"].value_counts().reset_index()
        sc.columns = ["Sentiment","Count"]
        st.dataframe(sc, use_container_width=True, hide_index=True)
    with col4:
        st.markdown("#### Top Actors by Movie Count")
        ac = movies["main_actor"].value_counts().head(8).reset_index()
        ac.columns = ["Actor","Movies"]
        st.dataframe(ac, use_container_width=True, hide_index=True)

    st.markdown("#### Data Quality Report")
    qdf = pd.DataFrame({
        "File":        ["movies","users","ratings","reviews"],
        "Rows":        [len(movies),len(users),len(ratings),len(reviews)],
        "Duplicates":  [movies.duplicated().sum(),users.duplicated().sum(),
                        ratings.duplicated(["user_id","movie_id"]).sum(),
                        reviews.duplicated(["user_id","movie_id"]).sum()],
        "Nulls":       [movies.isnull().sum().sum(),users.isnull().sum().sum(),
                        ratings.isnull().sum().sum(),reviews.isnull().sum().sum()],
        "Status":      ["Clean","Clean","Clean","Clean"],
    })
    st.dataframe(qdf, use_container_width=True, hide_index=True)

# ── User Recommendations ──────────────────────────────────────────────────────
elif page == "User Recommendations":
    st.markdown('<div class="section-hdr"> Personalised Recommendations</div>', unsafe_allow_html=True)
    st.caption("Collaborative filtering — user-user cosine similarity")

    col1, col2 = st.columns([1,2])
    with col1:
        uid   = st.number_input("User ID", 1, 500, 42)
        top_n = st.slider("Results", 3, 15, 8)
        run   = st.button("Get Recommendations", type="primary", use_container_width=True)
    with col2:
        urow = users[users.user_id == uid]
        if not urow.empty:
            r = urow.iloc[0]
            st.markdown(f"""
| Field | Value |
|---|---|
| Name | {r['name']} |
| Age | {r['age']} |
| Favourite Genre | {r['favorite_genre']} |
| Preferred Language | {r['preferred_language']} |
| Ratings given | {ratings[ratings.user_id==uid].shape[0]} |
            """)

    if run:
        recs = collab_recommend(uid, top_n, user_sim, mat, user_enc, prod_enc, movies)
        st.markdown(f"#### Top {top_n} picks for User {uid}")
        if recs.empty:
            st.info("Not enough data for this user.")
        else:
            for i,(_, row) in enumerate(recs.iterrows()):
                render_movie_card(row, rank=i+1)

# ── Similar Movies ────────────────────────────────────────────────────────────
elif page == "Similar Movies":
    st.markdown('<div class="section-hdr"> Similar Movies</div>', unsafe_allow_html=True)
    st.caption("Content-based filtering — TF-IDF on genres, actors, directors, description")

    selected = st.selectbox("Select a movie", movies["title"].tolist())
    top_n    = st.slider("Number of similar movies", 3, 12, 6)

    if st.button("Find Similar", type="primary"):
        mid  = int(movies[movies.title == selected]["movie_id"].values[0])
        recs = content_recommend(mid, top_n, prod_sim, mid_to_idx, movies)
        st.markdown("**Selected movie:**")
        render_movie_card(movies[movies.movie_id==mid].iloc[0])
        st.markdown(f"#### {top_n} similar movies")
        for i,(_, row) in enumerate(recs.iterrows()):
            render_movie_card(row, rank=i+1)

# ── Hybrid Mode ───────────────────────────────────────────────────────────────
elif page == " Hybrid Mode":
    st.markdown('<div class="section-hdr"> Hybrid Recommender</div>', unsafe_allow_html=True)
    st.caption("Blends collaborative filtering + content-based with tunable alpha")

    col1, col2 = st.columns(2)
    with col1:
        uid   = st.number_input("User ID", 1, 500, 10)
        top_n = st.slider("Results", 3, 12, 6)
    with col2:
        alpha = st.slider("CF weight (alpha)", 0.0, 1.0, 0.6, 0.05)
        st.caption("**1.0** = pure CF  |  **0.0** = pure content-based")

    if st.button("Generate Hybrid Recs", type="primary", use_container_width=True):
        recs = hybrid_recommend(uid, top_n, alpha, ratings,
                                user_sim, mat, user_enc, prod_enc,
                                prod_sim, mid_to_idx, movies)
        st.markdown(f"#### Hybrid recs for User {uid} (alpha={alpha})")
        for i,(_, row) in enumerate(recs.iterrows()):
            render_movie_card(row, rank=i+1)

# ── Evaluation ────────────────────────────────────────────────────────────────
elif page == "Evaluation":
    st.markdown('<div class="section-hdr">Model Evaluation</div>', unsafe_allow_html=True)

    K     = st.slider("K", 3, 15, 10)
    n_usr = st.slider("Test users", 20, 150, 80)

    if st.button("Run Evaluation", type="primary"):
        from sklearn.model_selection import train_test_split

        def p_at_k(r,rel,k): return len(set(r[:k])&set(rel))/k if k else 0
        def r_at_k(r,rel,k): return len(set(r[:k])&set(rel))/len(rel) if rel else 0
        def ndcg(r,rel,k):
            dcg  = sum(1/np.log2(i+2) for i,p in enumerate(r[:k]) if p in rel)
            idcg = sum(1/np.log2(i+2) for i in range(min(len(rel),k)))
            return dcg/idcg if idcg else 0

        _, test_r = train_test_split(ratings, test_size=0.2, random_state=42)
        p_s,r_s,n_s=[],[],[]
        prog = st.progress(0, text="Evaluating...")
        uids = test_r["user_id"].unique()[:n_usr]

        for idx_u, uid in enumerate(uids):
            rel = test_r[test_r.user_id==uid]["movie_id"].tolist()
            if not rel: continue
            recs_df  = collab_recommend(uid, K, user_sim, mat, user_enc, prod_enc, movies)
            rec_mids = recs_df["movie_id"].tolist()
            p_s.append(p_at_k(rec_mids,rel,K))
            r_s.append(r_at_k(rec_mids,rel,K))
            n_s.append(ndcg(rec_mids,rel,K))
            prog.progress((idx_u+1)/len(uids), text=f"Evaluated {idx_u+1}/{len(uids)} users")

        prog.empty()
        c1,c2,c3 = st.columns(3)
        kpi(c1, f"{np.mean(p_s):.3f}", f"Precision@{K}")
        kpi(c2, f"{np.mean(r_s):.3f}", f"Recall@{K}")
        kpi(c3, f"{np.mean(n_s):.3f}", f"NDCG@{K}")

        st.markdown("---")
        st.markdown("#### Score distributions")
        chart_df = pd.DataFrame({"Precision":p_s,"Recall":r_s,"NDCG":n_s})
        st.line_chart(chart_df)
        st.dataframe(chart_df.describe().round(4), use_container_width=True)

# ── Browse Movies ─────────────────────────────────────────────────────────────
elif page == "Browse Movies":
    st.markdown('<div class="section-hdr">Movie Catalog</div>', unsafe_allow_html=True)

    col1,col2,col3,col4 = st.columns(4)
    with col1:
        ind = st.selectbox("Industry", ["All"]+sorted(movies["industry"].unique()))
    with col2:
        lang = st.selectbox("Language", ["All"]+sorted(movies["language"].unique()))
    with col3:
        min_r = st.slider("Min Rating", 1.0, 5.0, 3.0, 0.1)
    with col4:
        sort_by = st.selectbox("Sort by", ["popularity_score","rating_avg","release_year"])

    filt = movies.copy()
    if ind  != "All": filt = filt[filt.industry == ind]
    if lang != "All": filt = filt[filt.language == lang]
    filt = filt[filt.rating_avg >= min_r].sort_values(sort_by, ascending=False)

    st.caption(f"Showing **{len(filt)}** of {len(movies)} movies")
    st.dataframe(
        filt[["movie_id","title","industry","language","genres","main_actor",
              "director","release_year","rating_avg","popularity_score"]],
        use_container_width=True, hide_index=True
    )

# ── Reviews ───────────────────────────────────────────────────────────────────
elif page == "Reviews":
    st.markdown('<div class="section-hdr"> User Reviews</div>', unsafe_allow_html=True)

    col1,col2 = st.columns(2)
    with col1:
        sent_filter = st.selectbox("Sentiment", ["All","positive","neutral","negative"])
    with col2:
        movie_filter = st.selectbox("Movie", ["All"]+movies["title"].tolist()[:50])

    rev = reviews.copy()
    if sent_filter  != "All": rev = rev[rev.sentiment == sent_filter]
    if movie_filter != "All":
        mid = int(movies[movies.title == movie_filter]["movie_id"].values[0])
        rev = rev[rev.movie_id == mid]

    rev_display = rev.merge(movies[["movie_id","title","main_actor"]], on="movie_id").head(30)
    st.caption(f"Showing {len(rev_display)} reviews")
    st.dataframe(
        rev_display[["review_id","user_id","title","sentiment","review_text"]],
        use_container_width=True, hide_index=True
    )

# ── API Docs ──────────────────────────────────────────────────────────────────
elif page == "🔗 API Docs":
    st.markdown('<div class="section-hdr">🔗 FastAPI Endpoints</div>', unsafe_allow_html=True)
    st.markdown("Start API: `uvicorn app.main:app --reload`")
    st.markdown("Swagger: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)")
    st.markdown("---")

    endpoints = [
        ("GET",  "/recommend/{user_id}",  "CF recommendations for a user"),
        ("GET",  "/similar/{movie_id}",   "Content-based similar movies"),
        ("GET",  "/hybrid/{user_id}",     "Hybrid recommendations"),
        ("GET",  "/movies",               "Browse movie catalog"),
        ("GET",  "/movies/{movie_id}",    "Movie detail"),
        ("GET",  "/users/{user_id}",      "User profile"),
        ("GET",  "/reviews/{movie_id}",   "Reviews for a movie"),
        ("GET",  "/health",               "API health check"),
    ]
    for method, path, desc in endpoints:
        color = "#dcfce7" if method=="GET" else "#fef9c3"
        tc    = "#166534" if method=="GET" else "#854d0e"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 14px;
                    background:white;border:1px solid #e5e7eb;border-radius:8px;margin-bottom:6px">
          <span style="background:{color};color:{tc};padding:3px 10px;
                       border-radius:6px;font-size:12px;font-weight:700">{method}</span>
          <code style="font-size:14px;color:#1d4ed8">{path}</code>
          <span style="font-size:13px;color:#6b7280">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.code("""# Get recs for user 42
curl http://127.0.0.1:8000/recommend/42?top_n=5

# Similar to movie 1
curl http://127.0.0.1:8000/similar/1?top_n=5

# Hybrid recs
curl http://127.0.0.1:8000/hybrid/42?top_n=6&alpha=0.7""", language="bash")
