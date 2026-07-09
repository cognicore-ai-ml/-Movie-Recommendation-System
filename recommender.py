import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Create a mock dataset (Replace this later with your actual dataset)
movies_data = {
    'movie_id': [1, 2, 3, 4],
    'title': ['The Dark Knight', 'Interstellar', 'Batman Begins', 'Inception'],
    'tags': [
        'action dc comics superhero batman dark gritty',
        'sci-fi space time travel Christopher Nolan',
        'action dc comics superhero batman origin',
        'sci-fi dreams heist mind-bending Christopher Nolan'
    ]
}

df = pd.DataFrame(movies_data)

# 2. Convert text tags into a matrix of token counts (Vectorization)
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['tags'])

# 3. Calculate the Cosine Similarity Matrix based on the formula
similarity = cosine_similarity(count_matrix)

# 4. Recommendation Function
def recommend(movie_title):
    # Find the index of the movie that matches the title
    movie_idx = df[df['title'] == movie_title].index[0]
    
    # Get pairwise similarity scores of all movies with that movie
    distances = similarity[movie_idx]
    
    # Sort the movies based on similarity scores
    # (returns tuples of (index, similarity_score))
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:]
    
    print(f"Recommendations for '{movie_title}':")
    for i in movies_list:
        print(f"- {df.iloc[i[0]]['title']} (Score: {round(i[1], 2)})")

# Test the system!
recommend('The Dark Knight')

