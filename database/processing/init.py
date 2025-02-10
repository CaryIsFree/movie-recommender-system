import sqlite3
import pandas as pd

def create_db():
    conn = sqlite3.connect('database/main.db')
    cursor = conn.cursor()
    
    # Table for the movielens dataset
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS movielens (
            id INTEGER PRIMARY KEY,
            title TEXT,
            release_date TEXT,
            vote_average INTEGER,
            vote_count INTEGER
        )
    ''')
    
    cursor.execute("SELECT id FROM movielens;")

    if len(cursor.fetchall()) == 0:
        df = pd.read_csv('data/movielens/movies_metadata.csv', low_memory=False)
    
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO movielens (title, release_date, vote_average, vote_count) 
                VALUES (?, ?, ?, ?)
            ''', (row['title'], row['release_date'], row['vote_average'], row['vote_count']))
    
    # Insert code to add netflix dataset here

    # Table for profile likes
    # Not adding a profile id because there will only be 1 profile
    # Feel free to add functionality for multiple users if you want
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS likes (
            movie_id INTEGER PRIMARY KEY
        )
    ''')  
    conn.commit()
    conn.close()

create_db()
