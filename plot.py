import numpy as np
import pandas as pd
import chess
import matplotlib.pyplot as plt

from regression import train_model
from features import board_features


def logistic_topk(board: chess.Board, model, k: int) -> np.ndarray:
    """Return the top-k destination squares predicted by the model."""
    x = board_features(board).reshape(1, -1)
    probs = model.predict_proba(x)[0]
    return np.argsort(probs)[-k:][::-1]


def eval_random_baseline(df_test: pd.DataFrame, trials: int = 50, k: int = 1, seed: int = 0) -> float:
    """
    Random baseline: pick random legal moves (k distinct moves per trial if possible).
    Returns estimated probability that true to-square is hit.
    """
    rng = np.random.default_rng(seed)
    hits = 0
    total = 0

    for fen, uci in zip(df_test["fen"], df_test["move_uci"]):
        board = chess.Board(fen)
        true_to = chess.Move.from_uci(uci).to_square
        legal = list(board.legal_moves)
        if len(legal) == 0:
            continue

        for _ in range(trials):

            sample_k = min(k, len(legal))
            idxs = rng.choice(len(legal), size=sample_k, replace=False)
            chosen_to = {legal[i].to_square for i in idxs}

            hits += int(true_to in chosen_to)
            total += 1

    return hits / total if total else 0.0


def eval_logistic(df_test: pd.DataFrame, model, k: int = 1) -> float:
    hits = 0
    total = 0

    for fen, uci in zip(df_test["fen"], df_test["move_uci"]):
        board = chess.Board(fen)
        true_to = chess.Move.from_uci(uci).to_square

        topk = set(logistic_topk(board, model, k=k))
        hits += int(true_to in topk)
        total += 1

    return hits / total if total else 0.0


def main():

    df = pd.read_csv("data/raw/Dr_Dragon_moves_dataset.csv")
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)


    X_train = []
    y_train = []

    for fen, uci in zip(df_train["fen"], df_train["move_uci"]):
        b = chess.Board(fen)
        m = chess.Move.from_uci(uci)
        X_train.append(board_features(b))
        y_train.append(m.to_square)

    X_train = np.vstack(X_train)
    y_train = np.array(y_train, dtype=np.int64)

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)
    model.fit(X_train, y_train)


    log_top1 = eval_logistic(df_test, model, k=1)
    log_top5 = eval_logistic(df_test, model, k=5)

    rand_top1 = eval_random_baseline(df_test, trials=50, k=1, seed=1)
    rand_top5 = eval_random_baseline(df_test, trials=50, k=5, seed=1)

    print("Random Top-1:", rand_top1)
    print("Logistic Top-1:", log_top1)
    print("Random Top-5:", rand_top5)
    print("Logistic Top-5:", log_top5)


    labels = ["Random Top-1", "Logistic Top-1", "Random Top-5", "Logistic Top-5"]
    values = [rand_top1, log_top1, rand_top5, log_top5]

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Logistic Regression vs Random Baseline (Destination-Square Accuracy)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
