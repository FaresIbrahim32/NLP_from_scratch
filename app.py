"""Interactive TF-IDF topic explorer for plain-text files."""

from __future__ import annotations

import re
from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

MIN_WORDS_PER_SEGMENT = 3
MIN_TOKEN_LENGTH = 2
TOP_TERMS_PER_SEGMENT = 3
MAX_VOCABULARY = 1500


st.set_page_config(
    page_title="Text Topic Explorer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def split_into_segments(text: str, mode: str, min_words: int) -> list[str]:
    """Split input into analyzable units while dropping tiny fragments."""
    if mode == "Paragraphs":
        raw_segments = re.split(r"\n\s*\n+", text)
    else:
        raw_segments = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in raw_segments if len(s.split()) >= min_words]


def clean_text(text: str, remove_stopwords: bool, min_token_length: int) -> str:
    """Lowercase and retain word-like tokens for TF-IDF."""
    tokens = [re.sub(r"^[^a-zA-Z]+|[^a-zA-Z'-]+$", "", token) for token in re.split(r"\s+", text.lower().strip())]
    tokens = [token for token in tokens if token]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    tokens = [t.strip("'-") for t in tokens if len(t.strip("'-")) >= min_token_length]
    return " ".join(t for t in tokens if t)


def analyze(
    text: str,
    split_mode: str,
    min_words: int,
    remove_stopwords: bool,
    min_token_length: int,
    max_features: int,
    ngram_range: tuple[int, int],
    top_terms: int,
) -> tuple[pd.DataFrame, pd.DataFrame, TfidfVectorizer]:
    segments = split_into_segments(text, split_mode, min_words)
    cleaned = [clean_text(s, remove_stopwords, min_token_length) for s in segments]
    usable = [(original, clean) for original, clean in zip(segments, cleaned) if clean]
    if not usable:
        raise ValueError("No usable text remained after preprocessing. Try lowering the filters.")

    original_segments, cleaned_segments = zip(*usable)
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(cleaned_segments)
    terms = np.array(vectorizer.get_feature_names_out())
    scores = matrix.toarray()

    rows = []
    for index, row in enumerate(scores, start=1):
        order = np.argsort(row)[::-1]
        ranked = [(terms[i], float(row[i])) for i in order[:top_terms] if row[i] > 0]
        rows.append(
            {
                "Segment": index,
                "Topic terms": ", ".join(term for term, _ in ranked),
                "TF-IDF strength": round(float(np.mean([score for _, score in ranked])), 4)
                if ranked
                else 0,
                "Original text": original_segments[index - 1],
                "Cleaned text": cleaned_segments[index - 1],
            }
        )
    segment_df = pd.DataFrame(rows)

    mean_scores = scores.mean(axis=0)
    order = np.argsort(mean_scores)[::-1]
    global_df = pd.DataFrame(
        {
            "Term": terms[order],
            "Average TF-IDF": mean_scores[order],
            "Segments containing term": (scores[:, order] > 0).sum(axis=0),
        }
    ).head(max(10, top_terms * 2))
    global_df["Average TF-IDF"] = global_df["Average TF-IDF"].round(4)
    return global_df, segment_df, vectorizer


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


st.title("🧭 Text Topic Explorer")
st.caption("Upload a plain-text file and discover the terms that best describe what it talks about.")

with st.sidebar:
    st.header("Analysis settings")
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    split_mode = st.radio("Analyze by", ["Paragraphs", "Sentences"], index=0)

if not uploaded_file:
    st.info("Upload a .txt file in the sidebar to begin.")
    st.markdown(
        """
        **What you’ll get**

        - A ranked set of global topic terms for the whole file
        - A chart showing which terms carry the most TF-IDF weight
        - Topic labels for each paragraph or sentence
        - A preview of the cleaned text used for analysis
        """
    )
    st.stop()

try:
    raw_text = uploaded_file.getvalue().decode("utf-8", errors="replace")
    global_df, segment_df, vectorizer = analyze(
        raw_text,
        split_mode,
        MIN_WORDS_PER_SEGMENT,
        True,
        MIN_TOKEN_LENGTH,
        MAX_VOCABULARY,
        (1, 1),
        TOP_TERMS_PER_SEGMENT,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

word_count = len(raw_text.split())
col1, col2, col3, col4 = st.columns(4)
col1.metric("Words", f"{word_count:,}")
col2.metric("Analyzed segments", len(segment_df))
col3.metric("Vocabulary", len(vectorizer.get_feature_names_out()))
col4.metric("Top topic", global_df.iloc[0]["Term"] if not global_df.empty else "—")

st.subheader("What this file is about")
topic_cols = st.columns(min(5, len(global_df)))
for column, (_, row) in zip(topic_cols, global_df.head(5).iterrows()):
    column.metric(row["Term"], f"{row['Average TF-IDF']:.3f}")

left, right = st.columns([1.1, 1])
with left:
    st.markdown("#### Most distinctive terms")
    chart_df = global_df.set_index("Term")["Average TF-IDF"].head(15)
    st.bar_chart(chart_df, horizontal=True)
with right:
    st.markdown("#### Ranked topic terms")
    st.dataframe(global_df, use_container_width=True, hide_index=True)

st.subheader(f"Topics by {split_mode[:-1].lower()}")
display_df = segment_df[["Segment", "Topic terms", "TF-IDF strength", "Original text"]]
st.dataframe(display_df, use_container_width=True, hide_index=True)

with st.expander("Inspect preprocessing"):
    st.caption("The cleaned text below is what TF-IDF sees after lowercasing, token filtering, and stopword removal.")
    st.dataframe(segment_df[["Segment", "Cleaned text"]], use_container_width=True, hide_index=True)

download_col1, download_col2 = st.columns(2)
download_col1.download_button(
    "Download topic results (CSV)",
    data=csv_bytes(segment_df),
    file_name="topic_results.csv",
    mime="text/csv",
    use_container_width=True,
)
download_col2.download_button(
    "Download ranked terms (CSV)",
    data=csv_bytes(global_df),
    file_name="ranked_topic_terms.csv",
    mime="text/csv",
    use_container_width=True,
)

