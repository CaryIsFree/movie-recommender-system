# Movie Recommender System Documentation

## Architecture

The system uses a content-based recommendation approach with the following components:

### 1. Data Processing
- Movie features are extracted from the MovieLens dataset
- Text features are processed using spaCy and sentence-transformers
- Numerical features are scaled using StandardScaler

### 2. Recommendation Engine
- Uses K-Nearest Neighbors for finding similar movies
- Features used include:
  - Movie genres (one-hot encoded)
  - Movie descriptions (text embeddings)
  - Popularity metrics
  - Vote counts and averages

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
