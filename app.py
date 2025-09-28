
import os
import time
import requests
import pandas as pd
import streamlit as st
import joblib
from datetime import datetime

# -----------------------------
# Streamlit page setup
# -----------------------------
st.set_page_config(page_title="Live News Sentiment (NewsAPI)", page_icon="📰", layout="wide")

# -----------------------------
# Model loading
# -----------------------------
@st.cache_resource
def load_model():
    path = os.getenv("SK_MODEL_PATH", "news_sentiment_sklearn.pkl")
    return joblib.load(path)

model = load_model()

# -----------------------------
# NewsAPI helpers
# -----------------------------
NEWS_API_KEY = os.getenv("NEWSAPI_API_KEY") or st.secrets.get("NEWSAPI_API_KEY", None)

BASE_TOP = "https://newsapi.org/v2/top-headlines"         # country/category/pageSize/page [web:76]
BASE_EVERYTHING = "https://newsapi.org/v2/everything"      # q/searchIn/sources/pageSize/sortBy/page [web:85]

HEADERS = {"X-Api-Key": NEWS_API_KEY} if NEWS_API_KEY else {}

def fetch_top_headlines(country="us", category=None, page_size=20, page=1):
    params = {
        "country": country,              # 2-letter ISO code; cannot combine with sources [web:76]
        "pageSize": min(int(page_size), 100),  # max 100 [web:76]
        "page": max(int(page), 1),
    }
    if category:
        params["category"] = category    # business/entertainment/general/health/science/sports/technology [web:76]
    r = requests.get(BASE_TOP, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    articles = data.get("articles", [])
    rows = []
    for a in articles:
        rows.append({
            "title": (a.get("title") or "").strip(),
            "description": (a.get("description") or "").strip(),
            "source": ((a.get("source") or {}).get("name") or "").strip(),
            "url": a.get("url") or "",
            "published_at": a.get("publishedAt") or "",
        })
    total = data.get("totalResults", len(rows))
    return pd.DataFrame(rows), total

def fetch_everything(q, page_size=20, page=1, language="en", search_in=None, sort_by="publishedAt"):
    # q supports advanced operators, must be URL-encoded by requests; searchIn can be title,description,content [web:85]
    params = {
        "q": q,
        "language": language,
        "pageSize": min(int(page_size), 100),  # max 100 [web:85]
        "page": max(int(page), 1),
        "sortBy": sort_by,                    # relevancy, popularity, publishedAt [web:85]
    }
    if search_in:
        params["searchIn"] = ",".join(search_in)  # e.g., ["title","description"] [web:85]
    r = requests.get(BASE_EVERYTHING, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    articles = data.get("articles", [])
    rows = []
    for a in articles:
        rows.append({
            "title": (a.get("title") or "").strip(),
            "description": (a.get("description") or "").strip(),
            "source": ((a.get("source") or {}).get("name") or "").strip(),
            "url": a.get("url") or "",
            "published_at": a.get("publishedAt") or "",
        })
    total = data.get("totalResults", len(rows))
    return pd.DataFrame(rows), total

def classify_titles(df):
    if df.empty:
        return df
    preds = model.predict(df["title"].fillna("").tolist())
    df["sentiment"] = ["Positive" if int(p) == 1 else "Negative" for p in preds]
    df["timestamp"] = datetime.utcnow().isoformat()
    return df

# -----------------------------
# UI
# -----------------------------
st.title("📰 Live News Sentiment (NewsAPI + TF‑IDF LR)")

if not NEWS_API_KEY:
    st.warning("Set NEWSAPI_API_KEY in environment or Streamlit secrets to enable live fetching. See NewsAPI docs for parameters and limits.")
    st.stop()

with st.sidebar:
    st.header("Fetch Options")
    mode = st.radio("Mode", ["Top Headlines", "Everything"])
    page_size = st.slider("Articles per call", 5, 100, 20, 5)  # NewsAPI max 100 [web:76][web:85]
    page = st.number_input("Page", min_value=1, value=1, step=1)

    if mode == "Top Headlines":
        country = st.selectbox("Country", ["us","in","gb","au","ca","de","fr","it","sg","za"], index=0)
        category = st.selectbox("Category", ["", "business","entertainment","general","health","science","sports","technology"], index=0)
    else:
        q = st.text_input("Query (supports advanced operators)", value="technology")  # quotes, +must, -not, AND/OR/NOT [web:85]
        language = st.selectbox("Language", ["en","de","fr","es","it","pt","ru","zh","ar"], index=0)
        search_in = st.multiselect("Search in fields", ["title","description","content"], default=["title","description"])
        sort_by = st.selectbox("Sort by", ["publishedAt","relevancy","popularity"], index=0)

    do_fetch = st.button("Fetch Live News")

# Fetch and classify
if do_fetch:
    try:
        if mode == "Top Headlines":
            df, total = fetch_top_headlines(country=country, category=(category or None), page_size=page_size, page=page)
        else:
            df, total = fetch_everything(q=q, page_size=page_size, page=page, language=language, search_in=search_in, sort_by=sort_by)

        df = classify_titles(df)

        st.success(f"Fetched {len(df)} of ~{total} results (page {page}).")
        pos = (df["sentiment"] == "Positive").sum()
        neg = (df["sentiment"] == "Negative").sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", len(df))
        c2.metric("Positive", pos)
        c3.metric("Negative", neg)

        # Filters
        with st.expander("Filters"):
            sel_sent = st.multiselect("Sentiment", options=["Positive","Negative"], default=["Positive","Negative"])
            sel_src = st.multiselect("Source", options=sorted(df["source"].dropna().unique().tolist()), default=None)
        fdf = df[df["sentiment"].isin(sel_sent)]
        if sel_src:
            fdf = fdf[fdf["source"].isin(sel_src)]

        st.dataframe(fdf[["title","sentiment","source","published_at","url"]], use_container_width=True)

        # Pagination hint (manual paging per NewsAPI docs)
        st.caption("Use the Page control in sidebar to navigate more results. NewsAPI returns up to pageSize per page, with max pageSize=100.")

    except requests.HTTPError as e:
        st.error(f"HTTP error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
