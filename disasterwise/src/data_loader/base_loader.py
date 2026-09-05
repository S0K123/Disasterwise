import os
import torch
from torch.utils.data import Dataset, DataLoader

class DisasterDataset(Dataset):
    """
    Base dataset class for disaster imagery.
    """
    def __init__(self, config, transform=None):
        self.config = config
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # TODO: Implement data discovery based on config['data']['xview_dir']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # TODO: Implement image loading and label parsing
        pass

def get_dataloader(config, mode='train', transform=None):
    """
    Creates a DataLoader for the specified mode.
    """
    dataset = DisasterDataset(config, transform=transform)
    batch_size = config['model']['parameters']['batch_size']
    shuffle = (mode == 'train')
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
