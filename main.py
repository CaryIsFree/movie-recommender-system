import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import ast
from sklearn.neighbors import NearestNeighbors
import spacy
import ast

# Load in the data using pandas
df = pd.read_csv('./data/movielens/movies_metadata.csv')

df_encoded = df.copy() # Copy the df so we preserve the original one
df_encoded = df_encoded[df_encoded['popularity'].apply(lambda x: isinstance(x, float))]
df_encoded = df_encoded[df_encoded['genres'].apply(lambda i:
    bool(ast.literal_eval(i)) and  # Ensure list is not empty
    isinstance(list(ast.literal_eval(i)[0].values())[1], str)  # Check second value type
)]
# Adult
df_encoded['adult'] = df_encoded['adult'].map({'True': 1, 'False': 0})

scaler = StandardScaler()

# Popularity
popularity = df_encoded['popularity']
X_normalized = scaler.fit_transform(popularity.to_numpy().reshape(-1, 1))
df_encoded['popularity'] = X_normalized

# Vote Average
vote_average = df_encoded['vote_average']
X_normalized = scaler.fit_transform(vote_average.to_numpy().reshape(-1, 1))
df_encoded['vote_average'] = X_normalized

# Vote Count
vote_count = df_encoded['vote_count']
X_normalized = scaler.fit_transform(vote_count.to_numpy().reshape(-1, 1))
df_encoded['vote_count'] = X_normalized

# Revenue
revenue = df_encoded['revenue']
X_normalized = scaler.fit_transform(revenue.to_numpy().reshape(-1, 1))
df_encoded['revenue'] = X_normalized

# Runtime
runtime = df_encoded['runtime']
X_normalized = scaler.fit_transform(runtime.to_numpy().reshape(-1, 1))
df_encoded['runtime'] = X_normalized

# Title
nlp = spacy.load("en_core_web_md")
def nlp_encode(x):
    res = nlp(x)
    return res.vector
df_encoded['title_vector'] = df_encoded['title'].astype(str).apply(nlp_encode)

# Genre
def parse_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres_list]
    except (ValueError, SyntaxError):
        return []

df_encoded['genres'] = df_encoded['genres'].apply(parse_genres)

mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(mlb.fit_transform(df_encoded['genres']), columns=mlb.classes_, index=df_encoded.index)

# Combine one-hot genres with original dataframe
df_encoded = pd.concat([df_encoded.drop(columns=['genres']), genre_encoded], axis=1)

# Description
# overview = df_encoded['overview']
# model = SentenceTransformer('all-MiniLM-L6-v2')
# embeddings = model.encode(overview)
# df_encoded['overview'] = df_encoded['overview'].apply(model.encode)
df_encoded = df_encoded.dropna().reset_index(drop=True)

# KNN for title search
title_vectors = np.array(df_encoded['title_vector'].tolist())
title_knn = NearestNeighbors(n_neighbors=1, metric='cosine')
title_knn.fit(title_vectors)

# KNN for recommendations
scaler = StandardScaler()
features = ['adult', 'popularity', 'vote_average', 'vote_count', 'revenue', 'runtime'] + list(mlb.classes_)
scaled_features = scaler.fit_transform(df_encoded[features])

rec_knn = NearestNeighbors(n_neighbors=3, metric='euclidean')
rec_knn.fit(scaled_features)

def recommend_movies(movie_title, num_recommendations=3):
    # Find the movie index using the title KNN
    movie_vector = nlp_encode(movie_title)
    distances, indices = title_knn.kneighbors([movie_vector])
    movie_idx = indices[0][0]

    # Get recommendations using the recommendation KNN
    distances, indices = rec_knn.kneighbors([scaled_features[movie_idx]], n_neighbors=num_recommendations+1)

    # Get recommended movie titles (excluding the input movie)
    recommended_titles = df_encoded.iloc[indices[0][1:]]['title'].tolist()
    return recommended_titles

print(recommend_movies('Curious George 2: Follow That Monkey!'))
