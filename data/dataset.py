import pandas as pd

from torch.utils.data import Dataset

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
        