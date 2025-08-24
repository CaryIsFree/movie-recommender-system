# Movie Recommender System Documentation

## Architecture

The system uses a content-based recommendation approach with the following components:

### 1. Data Processing
- Movie features are extracted from the MovieLens dataset
- Text processing:
  - Movie titles processed using spaCy (en_core_web_md model)
  - Movie descriptions processed using sentence-transformers (all-MiniLM-L6-v2 model)
- Genre processing using MultiLabelBinarizer for multi-label classification
- Numerical features normalized using StandardScaler

### 2. Recommendation Engine
- Uses K-Nearest Neighbors for finding similar movies
- Features used include:
  - Movie genres (multi-label binarized)
  - Movie titles (spaCy vector embeddings)
  - Movie descriptions (sentence transformer embeddings)
  - Numerical features (scaled):
    - Popularity
    - Vote counts and averages
    - Revenue
    - Runtime
    - Adult rating

### 3. Database
- SQLite database for storing:
  - Movie information
  - User likes
  - Movie metadata

### 4. Web Interface
- Flask web application
- Interactive search
- Like/unlike functionality
- Recommendation display

## API Endpoints

- `/` - Homepage showing liked movies
- `/search` - Movie search endpoint
- `/movie/<id>` - Individual movie page
- `/toggle_like` - Toggle movie like status
- `/profile` - User profile page
