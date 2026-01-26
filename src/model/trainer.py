class Trainer:
    """Handle model training"""
    - __init__(model, train_loader, val_loader, config)
    - train_epoch()
    - validate()
    - save_checkpoint(path)
    - load_checkpoint(path)
    - calculate_loss(predictions, targets, legal_moves_mask)
        # Cross-entropy with masking for illegal moves
    - calculate_metrics(predictions, targets)
        # Top-1, Top-3, Top-5 accuracy, perplexity