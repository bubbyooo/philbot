class BoardVisualizer:
    """Visualize the board and analysis"""
    - display_board(board, last_move=None)
    - display_move_probabilities(board, probs, top_n=5)
    - display_training_curves(history)
    - animate_game(game)
    - create_heatmap(attention_weights)  # If using attention