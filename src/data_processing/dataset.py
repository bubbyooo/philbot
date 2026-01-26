class ChessDataset(torch.utils.data.Dataset):
    """PyTorch dataset for training"""
    - __init__(positions, moves, context_features)
    - __getitem__(idx)
    - __len__()
    - collate_fn(batch)  # Custom batch collation