import torch
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pathlib import Path

def load_url_dataset(splits_directory: str, batch_size, train_ratio: float = 0.9, num_workers:int = 2, test: bool = False):
    transform = None

    splits_path = Path(splits_directory)

    assert splits_path.exists(), f"{splits_path} does not exist"
    assert splits_path.is_dir(), f"{splits_path} is not a directory"

    train_csv = splits_path / "train.csv"
    valid_csv = splits_path / "valid.csv"
    test_csv = splits_path / "test.csv"

    trainset = None
    if not test:
        trainset = UrlDataset(train_csv, transform=transform)

    testset = UrlDataset(test_csv, transform=transform)

    return _get_dataloaders(trainset, testset, batch_size, num_workers, train_ratio), testset.classes

def _get_dataloaders(trainset, testset, batch_size, num_workers, train_ratio):
    train_loader = None
    valid_loader = None
    if trainset:
        train_size = int(train_ratio * len(trainset))
        valid_size = len(trainset) - train_size
        trainset, validset = torch.utils.data.random_split(trainset, [train_size, valid_size])

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader

class UrlDataset(Dataset):
    def __init__(self, dataset_csv_file, transform=None):
        self.df = pd.read_csv(dataset_csv_file, index_col=0)
        self.transform = transform
        self.classes = list(self.df.iloc[:, 2].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        url = self.df.iloc[idx, 0]
        result = self.df.iloc[idx, 2]

        if self.transform:
            url = self.transform(url)

        return url, result
        