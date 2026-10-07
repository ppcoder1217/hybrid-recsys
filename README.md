# Hybrid Recommendation Engine with Cold-Start Handling

A movie recommender on **MovieLens 1M** that blends collaborative filtering (SVD) with a
content-based model (genres + decade). The blend weight **adapts to how much interaction
history a user and a movie have**, so the system degrades gracefully for new users and new movies
instead of breaking.

See [`WRITEUP.md`](WRITEUP.md) for the approach, key decisions, results, and failure analysis.

## Headline results

All numbers are top-10 ranking metrics (NDCG@10) on each user's 5 most recent ratings, held out
by time. A movie counts as relevant if the user rated it 4 or 5.

| Users | Popularity | CF only (SVD) | Content + popularity | **Hybrid** |
|---|---|---|---|---|
| Cold: 1 rating | 0.0159 | 0.0166 | 0.0154 | 0.0141 |
| Cold: 3 ratings | 0.0164 | 0.0191 | 0.0161 | **0.0200** |
| Cold: 5 ratings | 0.0169 | 0.0195 | 0.0171 | **0.0200** |
| Cold: 10 ratings | 0.0189 | 0.0263 | 0.0201 | 0.0259 |
| Warm: full history | 0.0357 | 0.0741 | 0.0427 | 0.0738 |

Cold **items** (150 movies with all ratings removed from training): the hybrid places a liked
cold movie in the top 10 of 150 with probability **0.152** vs **0.067** for random guessing.

Differences between methods in the cold rows are small and within noise for some rows; see the
write-up for an honest discussion.

## Setup

Requires Python 3.10+ (developed on 3.13).

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows (PowerShell)
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
```

On Windows, if `python` is not found, use `py -m venv venv`. If activation is blocked, run
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

## Data

1. Download **ml-1m.zip** from https://grouplens.org/datasets/movielens/1m/
2. Unzip it so the files sit at:

```
data/ml-1m/ratings.dat
data/ml-1m/movies.dat
data/ml-1m/users.dat
```

The `data/` folder is not committed to the repository.

## How to reproduce the results

Run from the repository root with the virtual environment active. Every random choice uses a
fixed seed, so the outputs are deterministic.

| Step | Command | What it does | Approx. time |
|---|---|---|---|
| 1 | `python explore.py` | Dataset statistics (users, items, sparsity, cold items) | seconds |
| 2 | `python data.py` | Builds the temporal train/test split and prints its sizes | seconds |
| 3 | `python baseline.py` | Popularity baseline on all users | ~1 min |
| 4 | `python cf.py` | SVD with 20 / 50 / 100 factors | ~2 min |
| 5 | `python cold_test.py` | CF vs popularity as history is truncated (1, 3, 5, 10 ratings) | a few min |
| 6 | `python content.py` | Content-based model, full and cold history | a few min |
| 7 | `python tune_content.py` | Tunes the popularity weight of the content model | a few min |
| 8 | `python tune_hybrid.py` | Tunes the blend parameter `k_user` on a separate user group | a few min |
| 9 | `python run_hybrid.py` | **Main results table**: popularity vs CF vs content vs hybrid | a few min |
| 10 | `python failure_analysis.py` | Failure analysis by history, item popularity, coverage, examples | ~2 min |
| 11 | `python cold_items.py` | Cold-**item** test (150 movies removed from training) | ~1 min |

### Re-running only the evaluation

The main table comes from `python run_hybrid.py`. To change the blend, edit the defaults of
`HybridRecommender` in `hybrid.py` (`k_user`, `k_item`, `n_factors`, `pop_weight`) and re-run it.
To evaluate a new model, write `recommend(user, seen_items, k) -> list_of_item_ids` and pass it to
`evaluate()` in `evaluate.py`.

## Project structure

```
data.py              load MovieLens, per-user temporal split
evaluate.py          Precision@K, Recall@K, NDCG@K
baseline.py          popularity baseline
cf.py                SVD collaborative filtering (scipy)
content.py           content-based model: genres + decade, popularity prior
hybrid.py            adaptive blend of CF and content
cold_test.py         cold-user simulation: CF vs popularity
tune_content.py      tunes the content model's popularity weight
tune_hybrid.py       tunes the blend parameter on separate users
run_hybrid.py        final comparison table
failure_analysis.py  where and why the hybrid fails
cold_items.py        cold-item evaluation
explore.py           dataset statistics
WRITEUP.md           approach, decisions, results, failure analysis
```

## Method in brief

- **Split:** each user's 5 most recent ratings are the test set; everything earlier is training.
  This avoids leaking the future into training.
- **CF:** truncated SVD (50 factors) on the user-item rating matrix; score = dot product of user
  and item factors.
- **Content:** each movie is a vector of genres and decade; a user's profile is the
  rating-weighted average of the movies they rated; score = cosine similarity plus a popularity prior.
- **Hybrid:** `score = a * z(CF) + (1 - a) * z(content)`, where
  `a = n_user / (n_user + 50) * n_item / (n_item + 5)`. A user or movie with no history gets zero
  CF weight, so content decides.
- **Cold-start slices:** cold users are simulated by keeping only the first 1, 3, 5 or 10 training
  ratings of 1,000 random users; cold items by removing all training ratings of 150 movies.

## Limitations

See the write-up. In short: the content features (genre, decade) are weak, differences between
methods in the cold-user rows are small, hyperparameters were tuned without a separate
validation split in places, and evaluation uses one split with no confidence intervals.
