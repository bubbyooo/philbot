class MoveSelector:
    """Select moves from model predictions"""
    - select_best_move(move_probs, legal_moves)
    - select_with_temperature(move_probs, legal_moves, temperature)
    - select_with_exploration(move_probs, legal_moves, epsilon)
    - apply_opening_book(board, opening_book)  # Use prof's openings