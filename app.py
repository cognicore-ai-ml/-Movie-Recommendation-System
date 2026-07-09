import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up the web page title
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie you like, and the system will instantly find similar matches using Cosine Similarity!")

# 1. Dataset setup
movies_data = {
    'movie_id': [1, 2, 3, 4, 5, 6],
    'title': ['The Dark Knight', 'Interstellar', 'Batman Begins', 'Inception', 'The Martian', 'The Prestige'],
    'tags': [
        'action dc comics superhero batman dark gritty trilogy christian bale',
        'sci-fi space time travel christopher nolan space exploration',
        'action dc comics superhero batman origin christian bale christopher nolan',
        'sci-fi dreams heist mind-bending christopher nolan leonardo dicaprio',
        'sci-fi space mars survival matt damon space exploration',
        'drama mystery magic illusion christopher nolan christian bale hugh jackman'
    ]
}
df = pd.DataFrame(movies_data)

# 2. Compute similarity vectors
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['tags'])
similarity = cosine_similarity(count_matrix)

# 3. User Interface
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown:",
    df['title'].values
)

if st.button("Get Recommendations"):
    # Find the index of the selected movie
    movie_idx = df[df['title'] == selected_movie].index[0]
    distances = similarity[movie_idx]
    
    # Sort and grab top 3 closest matches
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:4]
    
    st.write("---")
    st.subheader("🍿 Recommended For You:")
    
    for i in movies_list:
        match_percentage = round(i[1] * 100)
        st.success(f"🎥 **{df.iloc[i[0]]['title']}** — {match_percentage}% Match")
      
