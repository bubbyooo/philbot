class PositionEncoder:
    """Encode chess positions as tensors for neural network"""
    - encode_board(board) -> tensor
        # Multiple planes: pieces by type and color, castling rights,
        # en passant, repetition count, move count, etc.
    - encode_move(move, board) -> int  # Move as index
    - decode_move(move_index, board) -> chess.Move
    - get_legal_moves_mask(board) -> tensor  # Mask illegal moves