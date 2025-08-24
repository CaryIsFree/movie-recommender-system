from flask import Flask, request, render_template, jsonify
import sqlite3
import json
import joblib
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

# Load all ML models and features at startup
nlp = spacy.load("en_core_web_md")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load all models and features
df_encoded = joblib.load('../models/df_encoded.pkl')
scaled_numerical_features = joblib.load('../models/scaled_features.pkl')
title_knn = joblib.load('../models/title_knn.pkl')
rec_knn = joblib.load('../models/rec_knn.pkl')

# Create combined features for recommendations
overview_embeddings = np.array(df_encoded['overview_embedding'].tolist())
combined_features = np.concatenate([scaled_numerical_features, overview_embeddings], axis=1)

def get_db():
    conn = sqlite3.connect('../database/main.db')
    return conn

# Route for searching movies
@app.route('/search')
def search():
    query = request.args.get('q', '')
    conn = get_db() 
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, title FROM movielens WHERE title LIKE ? LIMIT 5", ('%' + query + '%',))
    movies = [{'id': row[0], 'title': row[1]} for row in cursor.fetchall()]
    
    conn.close()
    return jsonify(movies)

@app.route('/toggle_like', methods=['POST'])
def toggle_like():
    conn = get_db()
    cursor = conn.cursor()

    data = request.json
    movie_id = data.get('id')

    # Check if movie is already liked
    cursor.execute('SELECT * FROM likes WHERE movie_id = ?', (movie_id,))
    existing_like = cursor.fetchone()

    if existing_like:
        # Unlike: remove from likes
        cursor.execute('DELETE FROM likes WHERE movie_id = ?', (movie_id,))
        status = "unliked"
    else:
        # Like: add to likes
        cursor.execute('INSERT INTO likes (movie_id) VALUES (?)', (movie_id,))
        status = "liked"

    conn.commit()
    conn.close()
    return jsonify({"status": status})

@app.route('/clear_likes', methods=['POST'])
def clear_likes():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('DELETE from likes')
    conn.commit()
    conn.close()

@app.route('/')
def home():
    conn = get_db()
    cursor = conn.cursor()
    
    # Get only liked movies
    movies = cursor.execute('''
        SELECT DISTINCT m.* 
        FROM movielens m 
        INNER JOIN likes l ON m.id = l.movie_id
        ORDER BY m.title
    ''').fetchall()
    
    conn.close()
    return render_template('index.html', movies=movies)

@app.route('/profile')
def profile():
    conn = get_db()
    cursor = conn.cursor()

    ids = cursor.execute('SELECT movie_id FROM likes').fetchall() # Gets all of the profile's likes

    ids = [row[0] for row in ids]

    # Properly format the query with multiple placeholders
    if ids:
        placeholders = ','.join('?' * len(ids))  # Creates ?,?,? for parameterized query
        query = f'SELECT * FROM movielens WHERE id IN ({placeholders})'
        likes = cursor.execute(query, ids).fetchall()
    else:
        likes = [] 

    conn.close()

    return render_template('profile.html', likes=likes)

# Load all models and features
df_encoded = joblib.load('../models/df_encoded.pkl')
scaled_numerical_features = joblib.load('../models/scaled_features.pkl')
title_knn = joblib.load('../models/title_knn.pkl')
rec_knn = joblib.load('../models/rec_knn.pkl')

# Create combined features for recommendations
overview_embeddings = np.array(df_encoded['overview_embedding'].tolist())
combined_features = np.concatenate([scaled_numerical_features, overview_embeddings], axis=1)

def nlp_encode(x):
    res = nlp(x)
    return res.vector

def recommend_movies(movie_title, num_recommendations=3):
    try:
        # Find the movie index using the title KNN
        movie_vector = nlp_encode(movie_title)
        distances, indices = title_knn.kneighbors([movie_vector])
        movie_idx = indices[0][0]

        # Get recommendations using the recommendation KNN
        distances, indices = rec_knn.kneighbors([combined_features[movie_idx]], n_neighbors=num_recommendations*2)  # Get more recommendations as some might not be in DB

        # Get recommended movie titles (excluding the input movie)
        recommendations = []
        for idx in indices[0][1:]:  # Skip first one as it's the input movie
            movie = df_encoded.iloc[idx]
            # Verify that this movie exists in our database before adding it
            conn = get_db()
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM movielens WHERE title = ?', (movie['title'],))
            db_movie = cursor.fetchone()
            conn.close()
            
            if db_movie:  # Only add if movie exists in database
                recommendations.append({
                    'title': movie['title'],
                    'id': db_movie[0]  # Use the ID from our database
                })
            
            if len(recommendations) >= num_recommendations:  # Stop once we have enough valid recommendations
                break
                
        return recommendations
    except Exception as e:
        print(f"Error in recommend_movies: {e}")
        return []

@app.route('/movie/<int:movie_id>')
def movie_page(movie_id):
    try:
        conn = get_db()
        cursor = conn.cursor()

        # Try to get the movie info
        cursor.execute('SELECT * FROM movielens WHERE id = ?', (movie_id,))
        movie_info = cursor.fetchone()
        
        if movie_info is None:
            conn.close()
            return "Movie not found", 404
            
        # Check if movie is liked
        cursor.execute('SELECT * FROM likes WHERE movie_id = ?', (movie_id,))
        is_liked = cursor.fetchone() is not None
            
        movie_title = movie_info[1]  # Assuming title is the second column
        
        # Get actual recommendations based on the movie
        recommended_movies = recommend_movies(movie_title, num_recommendations=5)
        
        # Get full movie info for recommendations from database
        recommendations = []
        for rec in recommended_movies:
            movie = cursor.execute('SELECT * FROM movielens WHERE id = ?', (rec['id'],)).fetchone()
            if movie:  # If movie exists in database
                recommendations.append(movie)
        
        conn.close()
        return render_template('movie_page.html', movie_info=movie_info, recommendations=recommendations, is_liked=is_liked)
    
    except Exception as e:
        print(f"Error in movie_page: {e}")
        if 'conn' in locals():
            conn.close()
        return "An error occurred", 500


if __name__ == '__main__':
    app.run(debug=True)
