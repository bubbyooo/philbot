class ModelEvaluator:
    """Evaluate model performance"""
    - evaluate_on_held_out_games(model, games)
    - calculate_move_prediction_accuracy(model, positions, moves)
    - analyze_mistakes(model, games)
    - compare_to_baseline(model, random_baseline, stockfish_baseline)
    - generate_evaluation_report()