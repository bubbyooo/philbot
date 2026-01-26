class PhilBot:
    """Main bot class that plays chess using the trained model"""
    - __init__(model, position_encoder, config)
    - make_move(board, temperature=1.0)
    - get_move_probabilities(board)
    - set_thinking_time(seconds)
    - get_top_moves(board, n=3)
    - explain_move(board, move)  # Why this move was chosen