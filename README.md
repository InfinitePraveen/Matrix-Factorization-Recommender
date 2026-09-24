# Matrix Factorization Recommender

A lightweight movie recommendation system using SVD-based collaborative filtering. The project uses the MovieLens small dataset and the `Surprise` library to learn latent user/item factors from explicit ratings.

## Project Highlights

- SVD-based collaborative filtering
- MovieLens ratings dataset
- Lightweight CPU-only training
- Notebook-first workflow
- Flask web app for interactive recommendations
- Simple repository structure with no `src/` or separate preprocessing modules
- Model and metadata saved locally for the web app
- Includes contribution guidelines and changelog

## Repository Structure

```text
Matrix-Factorization-Recommender/
├── data/
│   └── README.md
├── models/
│   ├── README.md
│   └── svd_recommender.pkl
├── notebooks/
│   └── matrix_factorization_recommender.ipynb
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── app.py
├── CHANGELOG.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
└── requirements.txt
```

## Dataset

This project uses the open-source MovieLens small dataset from GroupLens. The notebook downloads the compact ZIP archive with Python `requests` and extracts only the required ratings and movie metadata files.

The dataset is not committed to the repository to keep the project small.

## How It Works

1. Download MovieLens data if it is not already available.
2. Load user, movie, and rating information.
3. Train a Surprise SVD collaborative-filtering model.
4. Evaluate the model with RMSE and MAE.
5. Generate recommendations for a selected user.
6. Save the trained model and movie metadata.
7. Run the Flask application and enter a user ID to see recommendations.

## Run the Notebook

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Open Jupyter:

```bash
jupyter notebook
```

Run `notebooks/matrix_factorization_recommender.ipynb` from top to bottom.

The notebook creates:

```text
models/svd_recommender.pkl
models/movies.csv
models/user_ids.csv
```

## Run the Web App

After running the notebook:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

Enter a MovieLens user ID to generate personalized recommendations.

## Technology

- Python
- Pandas
- NumPy
- Scikit-surprise
- Flask
- Jupyter Notebook
- MovieLens

## Profiles

GitHub: https://github.com/InfinitePraveen

LinkedIn: https://www.linkedin.com/in/infinitepraveen/

## License

This project is released under the MIT License.
