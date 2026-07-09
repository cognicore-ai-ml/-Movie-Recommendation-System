import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie you like, and the system will instantly find similar matches using Cosine Similarity!")

# 1. Load the dataset from the CSV file
df = pd.read_csv('movies.csv')

# 2. Compute similarity vectors (Vocabulary tracking)
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['tags'])

# 3. User Interface
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown:",
    df['title'].values
)

if st.button("Get Recommendations"):
    # Find the index of the selected movie
    movie_idx = df[df['title'] == selected_movie].index[0]
    
    # Extract only the single vector for our selected movie to save RAM
    selected_movie_vector = count_matrix[movie_idx]
    
    # Compute similarity for just this movie against all others on-the-fly
    similarity_scores = cosine_similarity(selected_movie_vector, count_matrix).flatten()
    
    # Sort and grab top 5 closest matches (excluding the movie itself)
    movies_list = sorted(list(enumerate(similarity_scores)), reverse=True, key=lambda x: x[1])[1:6]
    
    st.write("---")
    st.subheader("🍿 Recommended For You:")
    
    for i in movies_list:
        match_percentage = round(i[1] * 100)
        # Only show recommendations that have some keyword connection
        if match_percentage > 0:
            st.success(f"🎥 **{df.iloc[i[0]]['title']}** — {match_percentage}% Match")
            
