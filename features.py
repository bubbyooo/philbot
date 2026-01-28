import numpy as np
import chess

PIECE_TO_PLANE = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}
"""
This creates the features for the board. It creates 12 planes for the pieces, 
1 plane for the side to move, 4 planes for the castling rights, and 1 plane 
for the en passant square.
"""
def board_features(board: chess.Board) -> np.ndarray:
    x = np.zeros(12*64 + 1 + 4 + 1, dtype=np.float32)
    """
    This adds the pieces to the features for all the squares on the board and separates them by color.
    """
    for sq, piece in board.piece_map().items():
        offset = 0 if piece.color == chess.WHITE else 6
        plane = offset + PIECE_TO_PLANE[piece.piece_type]
        x[plane*64 + sq] = 1.0
        
    """
    This adds the side to move to the features.
    """
    i = 12*64
    x[i] = 1.0 if board.turn == chess.WHITE else 0.0
    i += 1

    """
    This adds the castling rights to the features.
    """
    x[i+0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    x[i+1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    x[i+2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    x[i+3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    i += 4

    """
    This adds the en passant square to the features.
    """
    x[i] = 1.0 if board.ep_square is not None else 0.0

    return x
