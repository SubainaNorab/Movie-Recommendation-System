
#  Movie Recommendation System

This is a simple movie recommendation system built using the MovieLens 100K dataset. It recommends movies based on collaborative filtering techniques and provides an interactive user interface with Streamlit.


## Features

- Recommends movies based on the selected title
- Predicts user ratings using user-based collaborative filtering (cosine similarity)
- Evaluates recommendation quality using Precision@K
- Clean and responsive Streamlit web interface


##  Project Structure

- `movie.py`: Core logic for loading data, building the model, making predictions, and evaluating precision  
- `movieUI.py`: Streamlit-based user interface that interacts with the backend logic  
- `README.md`: Project documentation


## Methodology

The system uses a user-item matrix built from the dataset, calculates cosine similarity between users, and predicts unseen ratings. It then ranks and recommends top movies based on predicted ratings. Precision@K is calculated to give a basic measure of recommendation quality.

## Technologies Used

- Python  
- Pandas & NumPy  
- Scikit-learn  
- Streamlit  
- MovieLens 100K Dataset




