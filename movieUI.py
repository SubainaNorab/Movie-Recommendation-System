import streamlit as st
import pandas as pd
from movie import recommend_movies, precision_at_k, user_item_matrix, user_similarity_df, movies, ratings

st.set_page_config(page_title="🎬 Movie Recommender", layout="centered")

# CSS Styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Movie Recommendation System")

# Top classic picks (manually chosen)
top_movies = [
    'Star Wars (1977)', 'Fargo (1996)', 'Toy Story (1995)',
    'Braveheart (1995)', 'Heat (1995)', 'Independence Day (ID4) (1996)',
    'Mission: Impossible (1996)', 'Scream (1996)', 'Rock (1996)'
]

# Get most-rated movies
top_rated_ids = ratings['item_id'].value_counts().head(30).index
popular_movies = movies[movies['movie_id'].isin(top_rated_ids)]['title'].tolist()

# Add some latest movies (e.g., released in 1995-1997)
latest_movies = movies[movies['title'].str.contains(r'\(199[5-7]\)', regex=True)]['title'].sample(10, random_state=42).tolist()

# Combine all options
movie_options = list(set(top_movies + popular_movies + latest_movies))
movie_options.sort()

# Movie selection
selected_movie = st.selectbox("🎞️ Choose a movie you like:", movie_options)

# Slider for number of recommendations
top_n = st.slider("🎯 How many recommendations do you want?", min_value=3, max_value=10, value=5)

# On button click
if st.button("Recommend"):
    with st.spinner("🔍 Finding similar movies..."):
        recommendations = recommend_movies(selected_movie, user_item_matrix, user_similarity_df, movies, top_n=top_n)
        prec = precision_at_k(selected_movie, user_item_matrix, user_similarity_df, movies, k=top_n)

        st.subheader("📌 Top Recommendations:")
        for title, score in recommendations:
            st.write(f"✅ **{title}** — Predicted Rating: {score:.2f}")

        st.markdown(f"📊 **Precision@{top_n}** = `{prec:.2f}`")
