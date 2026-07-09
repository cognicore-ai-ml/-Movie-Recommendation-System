import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load the dataset from the CSV file
df = pd.read_csv('movies.csv')

# 2. Vectorize the tags (This builds the vocabulary grid)
cv = CountVectorizer(stop_words='english')
count_matrix = cv.fit_transform(df['tags'])

# 3. Recommendation Function (Optimized for scale)
def recommend(movie_title):
    movie_title_clean = movie_title.strip().lower()
    titles_lower = df['title'].str.lower().str.strip()
    
    if movie_title_clean not in titles_lower.values:
        print(f"❌ Movie '{movie_title}' not found.")
        return
    
    # Find the index of the movie
    movie_idx = df[titles_lower == movie_title_clean].index[0]
    
    # Extract only the single vector for our selected movie to save RAM
    selected_movie_vector = count_matrix[movie_idx]
    
    # Compute similarity for just this movie against all others on-the-fly
    similarity_scores = cosine_similarity(selected_movie_vector, count_matrix).flatten()
    
    # Sort and grab top 5 closest matches (excluding the movie itself)
    movies_list = sorted(list(enumerate(similarity_scores)), reverse=True, key=lambda x: x[1])[1:6]
    
    print(f"🎬 Recommendations for '{df.iloc[movie_idx]['title']}':")
    for i in movies_list:
        match_percentage = round(i[1] * 100)
        if match_percentage > 0:
            print(f"  -> {df.iloc[i[0]]['title']} ({match_percentage}% Match)")

if __name__ == '__main__':
    print("--- Testing Scalable Movie Recommendation System ---\n")
    # Test with a movie from your new 200-movie list
    recommend('Interstellar')
    
