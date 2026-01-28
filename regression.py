import pandas as pd
import numpy as np
import chess

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score

from features import board_features

"""
This trains the model by reading the file, creating the features 
and doing a split.
It fits the model to the training data and evaluates it.
"""
def train_model(csv_path="data/raw/Dr_Dragon_moves_dataset.csv"):
    df = pd.read_csv(csv_path)

    X = []
    y = []

    for fen, uci in zip(df["fen"], df["move_uci"]):
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)

        
        y.append(move.to_square)
        X.append(board_features(board))

    X = np.vstack(X)
    y = np.array(y, dtype=np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    """
    This fits the model to the training data. We are using the logistic regression model with 2000 iterations
    and the lbfgs solver. We had to google which solver to use in order to get the L-BFGS solver. We could
    in the future use a CNN model instead.
    """
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1)

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print("Top-1 accuracy (to-square):", accuracy_score(y_test, pred))

    proba = clf.predict_proba(X_test)
    print("Top-5 accuracy (to-square):", top_k_accuracy_score(y_test, proba, k=5))

    return clf