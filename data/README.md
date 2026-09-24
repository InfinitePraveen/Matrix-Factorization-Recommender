# Data

This project uses the **MovieLens small dataset** provided by GroupLens.

The notebook downloads the dataset from the official MovieLens distribution using `requests` and extracts it locally.

The raw dataset is intentionally not committed to GitHub to keep the repository lightweight.

Expected local files after running the notebook:

```text
data/
├── ml-latest-small.zip
└── ml-latest-small/
    ├── movies.csv
    ├── ratings.csv
    └── ...
```

Dataset information: https://grouplens.org/datasets/movielens/
