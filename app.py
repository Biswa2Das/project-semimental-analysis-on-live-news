import os
import time
import json
import threading
import queue
import requests
import streamlit as st

from datetime import datetime, timezone
from typing import List, Dict

# Spark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

######################################################################
# Configuration
######################################################################

# Choose API provider: "newsapi" or "newsdata"
API_MODE = os.getenv("API_MODE", "newsapi")  # newsapi | newsdata
NEWSAPI_KEY = os.getenv("f6a082e3499c4779942ea3f429151fe3", "")
# Query terms for headlines
QUERY = os.getenv("QUERY", "technology")
LANG = os.getenv("LANG", "en")

# Streaming mode: "memory_queue" or "kafka"
STREAM_MODE = os.getenv("STREAM_MODE", "memory_queue")  # memory_queue | kafka

# Kafka parameters (if STREAM_MODE == "kafka")
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "news-headlines")

# Streamlit refresh seconds
DASHBOARD_REFRESH_SECS = int(os.getenv("DASHBOARD_REFRESH_SECS", "3"))

######################################################################
# Helper: News API fetchers
######################################################################

def fetch_news_newsapi(query: str, lang: str, page_size: int = 50) -> List[Dict]:
    """
    Uses NewsAPI.org /v2/everything to fetch recent articles.
    Requires NEWSAPI_KEY in env. Docs: https://newsapi.org/docs [web:2][web:20]
    """
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": lang,
        "pageSize": page_size,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    arts = data.get("articles", [])
    out = []
    for a in arts:
        out.append({
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "source": (a.get("source") or {}).get("name") or "",
            "url": a.get("url") or "",
            "publishedAt": a.get("publishedAt") or datetime.now(timezone.utc).isoformat()
        })
    return out

def fetch_news_newsdata(query: str, lang: str, size: int = 50) -> List[Dict]:
    """
    Uses NewsData.io /api/1/news to fetch recent articles.
    Requires NEWSDATA_API_KEY in env. Docs: https://newsdata.io [web:17]
    """
    if not NEWSDATA_API_KEY:
        return []
    url = "https://newsdata.io/api/1/news"
    params = {
        "apikey": NEWSDATA_API_KEY,
        "q": query,
        "language": lang,
        "page": 1
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", []) or []
    out = []
    for a in results[:size]:
        out.append({
            "title": a.get("title") or "",
            "description": a.get("description") or "",
            "source": a.get("source_id") or "",
            "url": a.get("link") or "",
            "publishedAt": a.get("pubDate") or datetime.now(timezone.utc).isoformat()
        })
    return out

def fetch_headlines(query: str, lang: str) -> List[Dict]:
    """
    Wrapper to fetch headlines based on API_MODE.
    """
    if API_MODE == "newsdata":
        return fetch_news_newsdata(query, lang)
    # default to newsapi
    return fetch_news_newsapi(query, lang)

######################################################################
# Producer Thread: Poll API and enqueue messages
######################################################################

class HeadlineProducer(threading.Thread):
    def __init__(self, q: queue.Queue, poll_interval: int = 10):
        super().__init__(daemon=True)
        self.q = q
        self.poll_interval = poll_interval
        self.seen = set()

    def run(self):
        while True:
            try:
                articles = fetch_headlines(QUERY, LANG)
                for art in articles:
                    key = (art.get("title") or "").strip()
                    if key and key not in self.seen:
                        self.seen.add(key)
                        payload = {
                            "title": art["title"],
                            "description": art["description"],
                            "source": art["source"],
                            "url": art["url"],
                            "publishedAt": art["publishedAt"],
                            "ingestedAt": datetime.utcnow().isoformat()
                        }
                        self.q.put(payload)
                time.sleep(self.poll_interval)
            except Exception:
                time.sleep(self.poll_interval)

######################################################################
# Spark Setup
######################################################################

def create_spark():
    """
    Create SparkSession with Kafka support if needed, per Spark + Kafka integration guide. [web:12]
    """
    builder = (
        SparkSession.builder
        .appName("RealTimeNewsSentiment")
        .config("spark.sql.streaming.schemaInference", "true")
    )
    # For local dev, enable single JVM
    builder = builder.master("local[*]")
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark

######################################################################
# Build a simple Sentiment Pipeline
######################################################################

def build_sentiment_pipeline():
    """
    Basic text -> tokens -> remove stopwords -> TF -> IDF -> LogisticRegression.
    Referencing common PySpark pipeline patterns for text classification. [web:7][web:16][web:13][web:10]
    """
    tokenizer = RegexTokenizer(inputCol="text", outputCol="tokens", pattern="\\W+")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered")
    tf = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1 << 18)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    # Label indexer expects string labels
    label_indexer = StringIndexer(inputCol="label_str", outputCol="label", handleInvalid="keep")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)
    pipe = Pipeline(stages=[tokenizer, remover, tf, idf, label_indexer, lr])
    return pipe

def seed_training_data(spark):
    """
    Tiny seed dataset for demonstration; replace with a proper labeled corpus for accuracy. [web:7][web:19]
    """
    samples = [
        ("Stocks rally to record highs after upbeat earnings", "Positive"),
        ("Company shares plunge amid revenue miss and layoffs", "Negative"),
        ("New product launch delights customers and boosts outlook", "Positive"),
        ("Regulatory probe triggers uncertainty for major bank", "Negative"),
        ("Tech giant posts stronger-than-expected profits", "Positive"),
        ("Supply chain disruptions hurt quarterly performance", "Negative"),
        ("Innovative partnership accelerates growth plans", "Positive"),
        ("Data breach exposes user information", "Negative"),
    ]
    df = spark.createDataFrame(samples, schema=["text", "label_str"])
    return df

def train_or_load_model(spark):
    train_df = seed_training_data(spark)
    pipe = build_sentiment_pipeline()
    model = pipe.fit(train_df)
    # Simple eval on training just to get a metric
    pred = model.transform(train_df)
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
    try:
        auc = evaluator.evaluate(pred)
        print(f"Seed AUC (train): {auc:.3f}")
    except Exception:
        pass
    return model

######################################################################
# Streaming: Memory queue source
######################################################################

def start_memory_queue_stream(spark, model, shared_table_name="news_predictions"):
    """
    Use a Python queue as source; micro-batch in a thread pushes rows to a Spark MemoryStream.
    We implement via foreachBatch on a static micro-batch created periodically. [web:12][web:15]
    """
    # Define schema for incoming records
    schema = StructType([
        StructField("title", StringType(), True),
        StructField("description", StringType(), True),
        StructField("source", StringType(), True),
        StructField("url", StringType(), True),
        StructField("publishedAt", StringType(), True),
        StructField("ingestedAt", StringType(), True)
    ])

    # Create rate source to trigger batches and use a thread-safe accumulator list for new items
    # Simpler: we poll a global queue each trigger, convert to DataFrame, union, then transform

    news_q = GLOBAL_NEWS_QUEUE

    def process_batch(_time, batch_df):
        # Drain queue
        items = []
        try:
            while True:
                items.append(news_q.get_nowait())
        except queue.Empty:
            pass

        if not items:
            return

        src_df = spark.createDataFrame(items, schema=schema)
        # Prepare text field for model
        df_prep = src_df.withColumn("text", F.coalesce(F.col("title"), F.lit("")))
        pred_df = model.transform(df_prep)
        out = (
            pred_df.select(
                "title", "source", "url",
                F.col("probability").alias("prob"),
                F.col("prediction").cast(IntegerType()).alias("prediction"),
                F.col("publishedAt").alias("published_at"),
                F.col("ingestedAt").alias("ingested_at")
            )
        )
        # Write to in-memory table for Streamlit
        out.write.format("memory").mode("append").saveAsTable(shared_table_name)

    # Dummy stream to trigger foreachBatch regularly
    rate_df = spark.readStream.format("rate").option("rowsPerSecond", 1).load()
    query = (
        rate_df.writeStream
        .outputMode("update")
        .foreachBatch(process_batch)
        .option("checkpointLocation", "./chk_memory_queue")
        .start()
    )
    return query

######################################################################
# Streaming: Kafka source (optional)
######################################################################

def start_kafka_stream(spark, model, shared_table_name="news_predictions"):
    """
    Read from Kafka topic and predict. Requires Spark Kafka integration and running Kafka. [web:12][web:6][web:18]
    """
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .load()
    )

    # Expect JSON messages with same fields as producer
    val_str = raw.selectExpr("CAST(value AS STRING) as json")
    schema = StructType([
        StructField("title", StringType(), True),
        StructField("description", StringType(), True),
        StructField("source", StringType(), True),
        StructField("url", StringType(), True),
        StructField("publishedAt", StringType(), True),
        StructField("ingestedAt", StringType(), True)
    ])

    parsed = val_str.select(F.from_json(F.col("json"), schema).alias("obj")).select("obj.*")
    df_prep = parsed.withColumn("text", F.coalesce(F.col("title"), F.lit("")))
    pred_df = model.transform(df_prep)
    out = (
        pred_df.select(
            "title", "source", "url",
            F.col("probability").alias("prob"),
            F.col("prediction").cast(IntegerType()).alias("prediction"),
            F.col("publishedAt").alias("published_at"),
            F.col("ingestedAt").alias("ingested_at")
        )
    )

    # Write to in-memory table for Streamlit
    query = (
        out.writeStream
        .format("memory")
        .queryName("pred_sink")
        .outputMode("append")
        .option("checkpointLocation", "./chk_kafka")
        .start()
    )
    return query

######################################################################
# Streamlit UI
######################################################################

def render_dashboard(spark, shared_table_name="news_predictions"):
    st.set_page_config(page_title="Real-Time News Sentiment", layout="wide")
    st.title("Real-Time News Sentiment")

    with st.sidebar:
        st.markdown(f"API: {API_MODE.upper()} | Query: {QUERY} | Lang: {LANG}")
        st.markdown(f"Stream: {STREAM_MODE}")
        st.markdown("Prediction: 1=Positive, 0=Negative")
        st.markdown("Auto-refresh enabled")

    # Auto-refresh
    st_autorefresh = st.empty()

    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("Latest Predictions")
        # Read from in-memory table
        try:
            df = spark.sql(f"SELECT *, CASE WHEN prediction=1 THEN 'Positive' ELSE 'Negative' END as sentiment FROM {shared_table_name} ORDER BY ingested_at DESC LIMIT 50")
            pdf = df.toPandas()
        except Exception:
            pdf = None

        if pdf is not None and not pdf.empty:
            pdf_view = pdf[["sentiment", "title", "source", "url", "published_at", "ingested_at"]]
            st.dataframe(pdf_view, use_container_width=True)
        else:
            st.info("Waiting for predictions...")

    with cols[1]:
        st.subheader("Counts (last 200)")
        try:
            agg = spark.sql(f"""
                SELECT
                    CASE WHEN prediction=1 THEN 'Positive' ELSE 'Negative' END as sentiment,
                    COUNT(*) as cnt
                FROM {shared_table_name}
                ORDER BY ingested_at DESC
                LIMIT 200
            """)
            agg_pdf = agg.groupBy("sentiment").agg(F.sum("cnt").alias("cnt")).toPandas()
        except Exception:
            agg_pdf = None

        if agg_pdf is not None and not agg_pdf.empty:
            st.bar_chart(agg_pdf.set_index("sentiment"))
        else:
            st.info("No counts yet.")

    time.sleep(DASHBOARD_REFRESH_SECS)
    st_autorefresh.write("")

######################################################################
# Main Orchestration
######################################################################

GLOBAL_NEWS_QUEUE = queue.Queue()

def main():
    # Start producer thread for memory_queue mode
    if STREAM_MODE == "memory_queue":
        producer = HeadlineProducer(GLOBAL_NEWS_QUEUE, poll_interval=10)
        producer.start()

    spark = create_spark()
    model = train_or_load_model(spark)

    shared_table_name = "news_predictions"

    if STREAM_MODE == "kafka":
        # To use Kafka in prod, publish fetched headlines with an external Kafka producer,
        # then enable this stream. Spark Kafka guide: spark.apache.org docs. [web:12][web:6][web:15]
        query = start_kafka_stream(spark, model, shared_table_name=shared_table_name)
    else:
        query = start_memory_queue_stream(spark, model, shared_table_name=shared_table_name)

    # Streamlit dashboard loop
    # Streamlit runs script top-to-bottom; use a simple loop guarded by session_state
    if "started" not in st.session_state:
        st.session_state["started"] = True

    # Render UI repeatedly; Streamlit reruns on interaction, so just render once
    render_dashboard(spark, shared_table_name=shared_table_name)

    # Keep query alive (Streamlit process typically lives)
    if query.isActive:
        pass

if __name__ == "__main__":
    main()
