# Movie Recommender System

A machine learning-based movie recommendation system that suggests movies based on user preferences and movie similarities.

## Prerequisites

While this application can run on various operating systems, this installation guide is specifically written for Linux users. Windows and macOS users may need to adapt the commands accordingly.

Before you begin, ensure you have the following installed:
- Python 3.13 or higher
- Visual Studio Code, PyCharm, or another Python IDE (recommended for development)
- Web browser (Chrome, Firefox, Safari, or Edge)

### System Requirements

- At least 4GB of RAM (8GB recommended)
- 2GB of free disk space for models and data
- Internet connection for initial setup and model downloads

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

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/CaryIsFree/movie-recommender-system.git
cd movie-recommender-system
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Open the project in your IDE:
- For VS Code: `code .`
- For PyCharm: Open the project directory

5. Configure Python Interpreter:
- In VS Code: Command Palette (Ctrl+Shift+P) → Python: Select Interpreter → Select the virtual environment
- In PyCharm: Settings → Project → Python Interpreter → Select the virtual environment

6. Generate the models (first time only):
```bash
python src/create_model.py
```

7. Run the application:
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
