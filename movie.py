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


def recommend_movies(selected_movie_title, user_item_matrix, user_similarity_df, movies, top_n=5):
    # Find users who rated this movie highly
    selected_movie_id = movies[movies['title'] == selected_movie_title]['movie_id'].values[0]
    high_raters = ratings[(ratings['item_id'] == selected_movie_id) & (ratings['rating'] >= 4)]['user_id']

    similar_users = user_similarity_df.loc[high_raters].mean().sort_values(ascending=False)

    # Compute weighted scores for unseen movies
    weighted_scores = {}
    for movie_id in user_item_matrix.columns:
        if movie_id == selected_movie_id:
            continue
        num, denom = 0, 0
        for other_user, sim_score in similar_users.items():
            if movie_id in user_item_matrix.columns and other_user in user_item_matrix.index:
                rating = user_item_matrix.loc[other_user, movie_id]
                if not pd.isna(rating):
                    num += sim_score * rating
                    denom += sim_score
        if denom > 0:
            weighted_scores[movie_id] = num / denom

    top_movies = sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [(movies[movies['movie_id'] == movie_id]['title'].values[0], score) for movie_id, score in top_movies]


def precision_at_k(selected_movie_title, user_item_matrix, user_similarity_df, movies, k=5, threshold=4):
    recommendations = recommend_movies(selected_movie_title, user_item_matrix, user_similarity_df, movies, top_n=k)
    recommended_ids = [movies[movies['title'] == title]['movie_id'].values[0] for title, _ in recommendations]
    selected_movie_id = movies[movies['title'] == selected_movie_title]['movie_id'].values[0]
    
    actual_users = ratings[(ratings['item_id'] == selected_movie_id) & (ratings['rating'] >= threshold)]['user_id']
    correct = 0
    for rid in recommended_ids:
        user_ratings = ratings[(ratings['item_id'] == rid) & (ratings['user_id'].isin(actual_users))]
        correct += len(user_ratings[user_ratings['rating'] >= threshold])
    
    return correct / k if k > 0 else 0
