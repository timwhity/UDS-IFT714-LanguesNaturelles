
from torch.utils.data import DataLoader
from pathlib import Path

from .dataset import UrlDataset
import torch
import pandas as pd

def balance_data(data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
    print("Balancing data...")
    
    # Make sure that the dataset has about the same number of benign and malicious samples
    num_benign = len(data[data["result"] == 0])
    num_malicious = len(data[data["result"] == 1])

    if num_benign > num_malicious:
        # Remove some benign samples
        indexes = data[data["result"] == 0].sample(n=num_benign - num_malicious, random_state=seed, replace=False).index
        data = data.drop(indexes)
        print("Dropped {} benign samples.".format(num_benign - num_malicious))
    elif num_malicious > num_benign:
        # Remove some malicious samples
        data = data.drop(data[data["result"] == 1].sample(n=num_malicious - num_benign, random_state=seed, replace=False).index)
        print("Dropped {} malicious samples.".format(num_malicious - num_benign))

    data.reset_index(inplace=True)

    print("Done.")
    return data

def load_url_dataset(splits_directory: str, batch_size, train_ratio: float = 0.9, num_workers:int = 2, test: bool = False):
    transform = None

    splits_path = Path(splits_directory)

    assert splits_path.exists(), f"{splits_path} does not exist"
    assert splits_path.is_dir(), f"{splits_path} is not a directory"

    train_csv = splits_path / "train.csv"
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

        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, test_loader

