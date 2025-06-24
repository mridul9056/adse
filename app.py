from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer

# Load data and model logic from movie_recommandation_system.py
# --- Data Preparation (copied and adapted from movie_recommandation_system.py) ---

def load_split_credits():
    credits1 = pd.read_csv("tmdb_5000_credits_1.csv")
    credits2 = pd.read_csv("tmdb_5000_credits_2.csv")
    return pd.concat([credits1, credits2], ignore_index=True)

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = load_split_credits()
df = movies.merge(credits, on='title')
movies = df[['id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace=True)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new_df = movies[['id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags'] = new_df['tags'].apply(stem)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# --- Recommendation Function ---
def recommend(movie):
    movie = movie.strip()
    if movie not in new_df['title'].values:
        return []
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    return [new_df.iloc[i[0]].title for i in movies_list]

# --- Flask App ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = None
    user_input = ''
    error = None
    if request.method == 'POST':
        user_input = request.form.get('movie')
        if user_input:
            recs = recommend(user_input)
            if recs:
                recommendations = recs
            else:
                error = f"No recommendations found. Please check the movie name or try another."
        else:
            error = "Please enter a movie name."
    return render_template('index.html', recommendations=recommendations, user_input=user_input, error=error)

if __name__ == '__main__':
    app.run(debug=True) 