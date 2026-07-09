# 🎬 Movie Recommendation System #

A machine learning-powered movie recommendation system that leverages Cosine Similarity to provide personalized movie suggestions based on content features. Built with Python and deployed with an interactive Streamlit interface.

# 🚀 Overview #

This project processes movie metadata (such as genres, keywords, cast, and crew) to calculate the similarity between films. When a user selects a movie, the system identifies the most closely related titles in the multi-dimensional feature space.

# 🛠 Tech Stack #

Language: Python
Data Analysis: pandas, numpy
Machine Learning: scikit-learn (for vectorization and cosine similarity)
Frontend: Streamlit

# 📋 Features #

Content-Based Filtering: Recommends movies similar to the one selected.
Vectorization: Uses CountVectorizer to convert text-based features into numerical data.
Interactive Dashboard: A clean, user-friendly UI to search and discover movie recommendations.

# ⚙️ How to Run Locally #

1.Clone the repository:
   ```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

```
2.Install the required libraries:

 ```bash
pip install -r requirements.txt
```

3.Launch the application:

```bash
streamlit run app.py
```

# 📈 Methodology #

The recommendation engine follows these steps:

Data Preprocessing: Cleaning and merging datasets to create a combined feature set.

Feature Engineering: Extracting relevant tags from genres, cast, and plot summaries.

Similarity Calculation: Applying the cosine similarity formula to measure the distance between movie vectors:

# \text{similarity} = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} #

# 📄 License #
This project is licensed under the MIT License - see the LICENSE file for details.
