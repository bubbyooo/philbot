class PGNParser:
    """Parse PGN files and extract all positions and moves"""
    - parse_file(pgn_path)
    - extract_positions_and_moves(game)
    - get_game_context(game)  # Opening name, time control, result
    - filter_games(min_elo=None, time_control=None)