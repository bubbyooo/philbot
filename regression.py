import pandas as pd
import numpy as np
import chess

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, top_k_accuracy_score
import tensorflow as tf, keras
from tensorflow.keras import layers

from features import board_features, board_features_cnn

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
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    print("Top-1 accuracy (to-square):", accuracy_score(y_test, pred))

    proba = clf.predict_proba(X_test)
    print("Top-5 accuracy (to-square):", top_k_accuracy_score(y_test, proba, k=5))

    return clf

def train_model_cnn(csv_path='data/raw/Dr_Dragon_moves_dataset.csv'):

    df = pd.read_csv(csv_path)

    X = []
    y = []

    for _, row in df.iterrows():
        board = chess.Board(row["fen"])
        board_feat = board_features_cnn(board)
        X.append(board_feat)

        move = chess.Move.from_uci(row["move_uci"])
        y.append(move.to_square)


    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = keras.Sequential([
    layers.Input(shape=(8, 8, 18)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(64, activation='softmax')  # 64 possible destination squares
])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

    return model
