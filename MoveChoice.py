import numpy as np
from features import board_features
import chess

def choose_move_logreg(board, model):
    """
    This chooses a move from the model's predictions based on the probability of the move's 
    destination square. It then sorts the legal moves by the probability of their destination 
    square and returns the move with the highest probability.
    """

    legal = list(board.legal_moves)
    if not legal:
        raise ValueError("No legal moves.")

    """
    This reshapes the board features into a 1D array and passes it to the model to 
    get the probabilities of the moves.
    """
    x = board_features(board).reshape(1, -1)
    probs = model.predict_proba(x)[0]

    scored = sorted(legal, key=lambda m: probs[m.to_square], reverse=True)

    """
    This returns the move with the highest probability destination square.
    """
    return scored[0]