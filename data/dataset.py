import pandas as pd

import torch
from torch.utils.data import Dataset

class UrlDataset(Dataset):
    def __init__(self, dataset_csv_file, transform=None, url_column=0, result_column=1):
        self.df = pd.read_csv(dataset_csv_file, index_col=0)
        self.transform = transform
        self.classes = [0, 1]

        self.url_column = url_column
        self.result_column = result_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        url = self.df.iloc[idx, self.url_column]
        result = self.df.iloc[idx, self.result_column]

        if self.transform:
            url = self.transform(url)

        return url, result
    
    def get_all_data(self):
        # Return the dataset as a tuple of numpy arrays
        return self.df.iloc[:, self.result_column].values, self.df.iloc[:, self.result_column].values
        