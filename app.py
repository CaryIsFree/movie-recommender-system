from flask import Flask, request, render_template, jsonify
import sqlite3
import json

app = Flask(__name__)

def get_db():
    conn = sqlite3.connect('database/main.db')
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

@app.route('/add_like', methods=['POST'])
def add_like():
    conn = get_db()
    cursor = conn.cursor()

    data = request.json
    page_id = data.get('id')

    cursor.execute('INSERT INTO likes (movie_id) VALUES (?)', (page_id,))
    conn.commit()
    conn.close()

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
    movies = cursor.execute('SELECT * FROM movielens ORDER BY RANDOM() LIMIT 15').fetchall() # Gets 15 random movies
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

@app.route('/movie/<int:movie_id>')
def movie_page(movie_id):
    conn = get_db()
    cursor = conn.cursor()

    movie_info = cursor.execute('SELECT * FROM movielens WHERE id = ?', (movie_id,)).fetchall()[0] # Gets selected movie's info
    recommendations = cursor.execute('SELECT * FROM movielens ORDER BY RANDOM() LIMIT 5').fetchall() # Gets 5 random 'recommendations'

    return render_template('movie_page.html', movie_info=movie_info, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
