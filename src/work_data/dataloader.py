import torch
from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(dataset, batch_size=4, shuffle=True, num_workers=4):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
