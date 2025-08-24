# Movie Recommender System

A machine learning-based movie recommendation system that suggests movies based on user preferences and movie similarities.

## Features

- Content-based movie recommendations
- User liked movies tracking
- Movie search functionality
- Interactive web interface
- Recommendation engine using KNN

## Tech Stack

- Python 3.13
- Flask
- scikit-learn
- spaCy
- sentence-transformers
- SQLite

## Project Structure

```
movie-recommender-system/
├── src/                     # Source code
│   ├── models/             # ML models and data processing
│   ├── static/             # Frontend assets
│   └── templates/          # HTML templates
├── docs/                   # Documentation
├── data/                   # Dataset files
│   └── movielens/         # MovieLens dataset
├── models/                 # Saved ML models
├── tests/                  # Unit tests
└── requirements.txt        # Dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python src/app.py
```

## Usage

1. Visit `http://localhost:5000` in your web browser
2. Search for movies in the search bar
3. Like movies to see recommendations
4. View your liked movies on the homepage

## Data

This project uses the MovieLens dataset for movie recommendations. The dataset includes:
- Movie metadata
- User ratings
- Movie genres
