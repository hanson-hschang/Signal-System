from typing import List, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset, random_split


def dataset_split_to_loaders(
    dataset: Dataset,
    split_ratio: Sequence[float],
    batch_size: int = 128,
    shuffle: bool = True,
    random_seed: Optional[int] = None,
) -> List[DataLoader]:
    generator = (
        torch.Generator().manual_seed(random_seed) if random_seed else None
    )
    datasets = random_split(
        dataset=dataset,
        lengths=split_ratio,
        generator=generator,
    )
    data_loaders = []
    for dataset in datasets:
        data_loaders.append(
            DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        )
    return data_loaders
