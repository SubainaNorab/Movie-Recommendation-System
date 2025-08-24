import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load ratings
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Load movie titles
movies = pd.read_csv(
    'ml-100k/u.item',
    sep='|',
    encoding='latin-1',
    names=['movie_id', 'title'] + [str(i) for i in range(22)],
    usecols=[0, 1]
)

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='user_id', columns='item_id', values='rating')
user_item_filled = user_item_matrix.fillna(0)

# Compute cosine similarity
user_similarity = cosine_similarity(user_item_filled)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


def recommend_movies(selected_movie_title, user_item_matrix, user_similarity_df, movies, top_n=5, threshold=4):
    # Get movie id
    selected_movie_id = movies[movies['title'] == selected_movie_title]['movie_id'].values[0]

    # Users who rated this movie >= threshold
    high_raters = ratings[(ratings['item_id'] == selected_movie_id) & (ratings['rating'] >= threshold)]['user_id']

    if high_raters.empty:
        return []

    # Compute similarity scores across high raters
    similar_users = user_similarity_df.loc[high_raters].mean()

    # Weighted prediction for all movies (vectorized)
    weighted_scores = user_item_matrix.T.dot(similar_users) / (similar_users.sum() + 1e-9)

    # Drop the selected movie from recommendations
    weighted_scores = weighted_scores.drop(selected_movie_id, errors="ignore")

    # Top-N movies
    top_movies = weighted_scores.sort_values(ascending=False).head(top_n)

    return [(movies[movies['movie_id'] == mid]['title'].values[0], score) for mid, score in top_movies.items()]


def precision_at_k(selected_movie_title, user_item_matrix, user_similarity_df, movies, k=5, threshold=4):
    recommendations = recommend_movies(selected_movie_title, user_item_matrix, user_similarity_df, movies, top_n=k)
    if not recommendations:
        return 0.0

    recommended_ids = [movies[movies['title'] == title]['movie_id'].values[0] for title, _ in recommendations]
    selected_movie_id = movies[movies['title'] == selected_movie_title]['movie_id'].values[0]

    # Actual relevant users
    actual_users = ratings[(ratings['item_id'] == selected_movie_id) & (ratings['rating'] >= threshold)]['user_id']

    correct = 0
    for rid in recommended_ids:
        user_ratings = ratings[(ratings['item_id'] == rid) & (ratings['user_id'].isin(actual_users))]
        correct += len(user_ratings[user_ratings['rating'] >= threshold])

    return correct / k if k > 0 else 0
