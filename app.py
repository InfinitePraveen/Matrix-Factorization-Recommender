from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "svd_recommender.pkl"
MOVIES_PATH = BASE_DIR / "models" / "movies.csv"
USER_IDS_PATH = BASE_DIR / "models" / "user_ids.csv"

app = Flask(__name__)

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
movies = pd.read_csv(MOVIES_PATH) if MOVIES_PATH.exists() else pd.DataFrame()
user_ids = (
    pd.read_csv(USER_IDS_PATH)["userId"].astype(int).tolist()
    if USER_IDS_PATH.exists()
    else []
)


def recommend_movies(user_id, number_of_recommendations=10):
    if model is None or movies.empty:
        return []

    if user_id not in user_ids:
        return []

    rated = model.trainset.ur[model.trainset.to_inner_uid(user_id)]
    rated_movie_ids = {
        model.trainset.to_raw_iid(inner_movie_id)
        for inner_movie_id, _ in rated
    }

    predictions = []
    for movie_id in movies["movieId"].tolist():
        if movie_id in rated_movie_ids:
            continue

        prediction = model.predict(user_id, int(movie_id))
        predictions.append((movie_id, prediction.est))

    predictions.sort(key=lambda item: item[1], reverse=True)

    movie_lookup = movies.set_index("movieId")["title"].to_dict()
    return [
        {"title": movie_lookup[movie_id], "score": round(score, 2)}
        for movie_id, score in predictions[:number_of_recommendations]
    ]


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    message = ""
    entered_user_id = ""

    if request.method == "POST":
        entered_user_id = request.form.get("user_id", "").strip()

        try:
            user_id = int(entered_user_id)
            recommendations = recommend_movies(user_id)

            if not recommendations:
                message = (
                    "User ID was not found. Try a valid MovieLens user ID "
                    "from the dataset."
                )
        except ValueError:
            message = "Please enter a numeric MovieLens user ID."

    return render_template(
        "index.html",
        recommendations=recommendations,
        message=message,
        user_id=entered_user_id,
        total_users=len(user_ids),
    )


if __name__ == "__main__":
    app.run(debug=True)
