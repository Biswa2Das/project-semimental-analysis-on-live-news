import streamlit as st
import requests
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

st.set_page_config(page_title="Kerala News Sentiment Dashboard", layout="wide")

# ---- API Setup ----
API_KEY = "pub_90c7e03f76c44364828a85c437caa6c9"
URL = f"https://newsdata.io/api/1/news?apikey={API_KEY}&q=Kerala&country=in&language=en"

# ---- Fetch News ----
@st.cache_data(ttl=600)
def fetch_news():
    res = requests.get(URL)
    data = res.json()
    articles = data.get("results", [])
    df = pd.DataFrame(articles)[["title", "description", "pubDate", "link"]]
    return df

df = fetch_news()
st.title("📰 Kerala News Sentiment Dashboard")
st.caption("Live news from NewsData.io with real-time sentiment analysis")

# ---- Sentiment Analysis ----
def get_sentiment(text):
    if text:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"
    return "Unknown"

df["sentiment"] = df["description"].apply(get_sentiment)

# ---- Pie Chart ----
sentiment_counts = df["sentiment"].value_counts()
fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["green", "red", "gray"])
ax.set_title("Overall Sentiment Distribution")
st.pyplot(fig)

# ---- News Selector ----
selected_title = st.selectbox("🗞️ Pick a news headline to analyze:", df["title"])
selected_article = df[df["title"] == selected_title].iloc[0]

st.subheader(selected_article["title"])
st.write(selected_article["description"])
st.markdown(f"🔗 [Read Full Article]({selected_article['link']})")

# ---- Detailed Sentiment ----
blob = TextBlob(selected_article["description"] or "")
sentiment = blob.sentiment
st.markdown(f"**Polarity Score:** `{sentiment.polarity}` (−1 to +1)")
st.markdown(f"**Subjectivity Score:** `{sentiment.subjectivity}` (0 = Fact, 1 = Opinion)")

if sentiment.polarity > 0:
    st.success("This news is **Positive** 🟢")
elif sentiment.polarity < 0:
    st.error("This news is **Negative** 🔴")
else:
    st.info("This news is **Neutral** ⚪")
