import streamlit as st
import joblib
import gdown
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from difflib import get_close_matches

# ============ Google Drive Links ============
PIVOT_TABLE_URL = "https://drive.google.com/uc?id=1u50YACOIyFtgruj7VH2XQ3TvWqF6y-QT"
MODEL_URL = "https://drive.google.com/uc?id=1EK5Gueb-EDGFGg6RvKhQAAfdDqsTEWRd"

# ============ Download Helper ============
def download_file(url, local_name):
    if not os.path.exists(local_name):
        with st.spinner(f"⬇️ Downloading {local_name} ... please wait"):
            gdown.download(url, local_name, quiet=False, fuzzy=True)
    return local_name

# ============ Cached Loader ============
@st.cache_resource
def load_data():
    pivot_file = download_file(PIVOT_TABLE_URL, "pivot_table.joblib")
    model_file = download_file(MODEL_URL, "book_recommender_model.joblib")

    pivot_table = joblib.load(pivot_file)
    model = joblib.load(model_file)

    return pivot_table, model

# ⭐ FIX: actually load the data here
pivot_table, model = load_data()

# ============ Page Config ============
st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="centered")

# ============ Sidebar ============
with st.sidebar:
    st.markdown("<h2 style='color:#4B6EA9;'>📚 Book Recommender</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size: 15px;'>
        Looking for your next great read? ✨<br><br>
        Just type the name of a book you love, and we’ll suggest others you might enjoy — 
        based on what readers like you also appreciated.
    </p>
    <hr style='border-top: 1px solid #bbb;'>
    <p style='font-size: 13px; color:#666;'>
        Smart, simple, and tailored to your taste. Start exploring!
    </p>
    """, unsafe_allow_html=True)

# ============ Title ============
st.markdown("<h1 style='text-align: center; color: #6C63FF;'>📚 Book Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Find your next favorite read in seconds</p>", unsafe_allow_html=True)
st.markdown("---")

# ============ Input ============
book_name = st.text_input("🔍 Enter a book you like:")
n_recommendations = st.slider("📎 Number of recommendations", 1, 10, 5)

# ============ Recommend Button ============
if st.button("✨ Recommend"):
    book_query = book_name.strip()

    if book_query not in pivot_table.columns:
        matches = get_close_matches(book_query, pivot_table.columns, n=1)
        if matches:
            corrected_name = matches[0]
            st.warning(f"Book not found. Did you mean: **{corrected_name}**?")
            book_query = corrected_name
        else:
            st.error("❌ Book not found in database.")
            st.stop()

    # Get recommendations
    book_vector = pivot_table[book_query].values.reshape(1, -1)
    distances, indices = model.kneighbors(book_vector, n_neighbors=n_recommendations + 1)

    st.markdown(f"<h3 style='color:#4CAF50;'>📚 Because you liked: <em>{book_query}</em></h3>", unsafe_allow_html=True)
    st.markdown("### You might also enjoy:")

    count = 0
    for i in indices.flatten():
        similar_book = pivot_table.columns[i]
        if similar_book != book_query and count < n_recommendations:
            st.markdown(f"📘 **{similar_book}**")
            count += 1
