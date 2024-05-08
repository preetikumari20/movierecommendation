from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import os

app = Flask(__name__)

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the dataset
df = pd.read_csv(os.path.join(BASE_DIR, 'dataset.csv'))

# Function to preprocess the data
def preprocess_data(x):
    return str(x['title']) + ' ' + str(x['genre']) + ' ' + str(x['popularity']) + ' ' + str(x['vote_count'])

# Apply the preprocessing function to the dataset
df['combined_features'] = df.apply(preprocess_data, axis=1)

# Create a CountVectorizer object
cv = CountVectorizer()

# Fit and transform the data
count_matrix = cv.fit_transform(df['combined_features'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix)

# Function to get movie recommendations
def get_recommendations(movie):
    idx = df[df['title'] == movie].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return list(df['title'].iloc[movie_indices])

@app.route('/')
def index():
    # Get the list of movie titles sorted alphabetically
    movies = sorted(df['title'].tolist())
    return render_template('index.html', movies=movies)

@app.route('/recommend', methods=['POST'])
def recommend():
    movie = request.form['movie']
    recommendations = get_recommendations(movie)
    # Get the list of movie titles sorted alphabetically
    movies = sorted(df['title'].tolist())
    return render_template('index.html', movie=movie, recommendations=recommendations, movies=movies)

if __name__ == '__main__':
    app.run(debug=True)
