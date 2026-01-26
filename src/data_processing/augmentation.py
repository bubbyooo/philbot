class DataAugmenter:
    """Augment limited data with symmetries and variations"""
    - horizontal_flip(board, move)
    - add_noise_to_position(board, epsilon)  # Slightly perturb features
    - augment_dataset(positions, moves, factor=2)