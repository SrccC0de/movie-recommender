import streamlit as st
import pandas as pd

# Load dataset
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv("u.data", sep='\t', names=column_names)

# Load movie titles
movie_titles = pd.read_csv("u.item", sep='|', encoding='latin-1', header=None, usecols=[0, 1], names=['item_id', 'title'])

# Merge datasets
movie_data = pd.merge(df, movie_titles, on='item_id')

# Create ratings summary
ratings_summary = movie_data.groupby('title').agg({'rating': ['mean', 'count']})
ratings_summary.columns = ['mean_rating', 'num_ratings']

# Create user-movie matrix
user_movie_matrix = movie_data.pivot_table(index='user_id', columns='title', values='rating')

# Function to get similar movies
def get_similar_movies(movie_name, min_ratings=100):
    if movie_name not in user_movie_matrix.columns:
        return f"Movie '{movie_name}' not found in dataset."

    target_movie_ratings = user_movie_matrix[movie_name]
    similar_scores = user_movie_matrix.corrwith(target_movie_ratings)

    corr_df = pd.DataFrame(similar_scores, columns=['correlation'])
    corr_df.dropna(inplace=True)
    corr_df = corr_df.join(ratings_summary['num_ratings'])
    
    recommendations = corr_df[corr_df['num_ratings'] > min_ratings].sort_values('correlation', ascending=False)
    
    return recommendations[recommendations.index != movie_name].head(10)

# Streamlit UI
st.title('Movie Recommender System')

# Get user input for movie title
movie_name = st.selectbox("Choose a Movie:", sorted(user_movie_matrix.columns))

# Display recommendations without posters
if movie_name:
    recommendations = get_similar_movies(movie_name)
    st.write(f"Top 10 Movies similar to **{movie_name}**:")
    
    # Show the movie names and their similarity scores
    for index, row in recommendations.iterrows():
        st.write(f"{index} - Correlation: {row['correlation']} (Ratings: {row['num_ratings']})")
