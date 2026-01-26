class OpeningBook:
    """
    Build opening book from professor's games
    Use first N moves from his games as preferred openings
    """
    - __init__(games)
    - build_book(max_depth=10)
    - get_move(board)  # Returns move if in book, else None
    - get_statistics(position)  # How often prof played what