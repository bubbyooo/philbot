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

def board_features_cnn(board: chess.Board) -> np.ndarray:
    """
    Creates an 8x8x18 feature representation for CNN:
    - 6 planes for white pieces (pawn, knight, bishop, rook, queen, king)
    - 6 planes for black pieces
    - 1 plane for side to move
    - 4 planes for castling rights
    - 1 plane for en passant
    """
    features = np.zeros((8, 8, 18), dtype=np.float32)
    
    # Planes 0-11: Piece positions (6 white + 6 black)
    for sq, piece in board.piece_map().items():
        rank = sq // 8
        file = sq % 8
        offset = 0 if piece.color == chess.WHITE else 6
        plane = offset + PIECE_TO_PLANE[piece.piece_type]
        features[rank, file, plane] = 1.0
    
    # Plane 12: Side to move (1 for white, 0 for black)
    features[:, :, 12] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Planes 13-16: Castling rights (broadcast across entire board)
    features[:, :, 13] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    features[:, :, 14] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    features[:, :, 15] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    features[:, :, 16] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    # Plane 17: En passant square
    if board.ep_square is not None:
        ep_rank = board.ep_square // 8
        ep_file = board.ep_square % 8
        features[ep_rank, ep_file, 17] = 1.0
    
    return features