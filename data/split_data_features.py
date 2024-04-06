import argparse
import pandas as pd
from pathlib import Path
from stats_dataset import get_dataset_stats, plot_dataset_stats
from data_utils import balance_data

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing data...", end="")
    
    # Rename column of data from "phishing" to "result"
    data.rename(columns={"phishing": "result"}, inplace=True)

    print("Done.")
    return data

def main(args):
    dataset_file = args.dataset
    save_dir = Path(args.dataset).parent
    test_ratio = args.test_ratio
    seed = args.seed

    print("Reading & Preprocessing ", dataset_file, "...", end="")

    if not Path(dataset_file).exists():
        raise FileNotFoundError(f"Dataset file {dataset_file} not found.")
    
    df = pd.read_csv(dataset_file)
    df = preprocess_data(df)

    if args.balance:
        df = balance_data(df, seed)

    print("Splitting data...")
    # Split the data into training, validation, and testing sets
    test = df.sample(frac=test_ratio, random_state=seed)
    train = df.drop(test.index)

    if not save_dir.exists():
        save_dir.mkdir()

    splits_save_dir = save_dir / "splits"
    if not splits_save_dir.exists():
        splits_save_dir.mkdir()

    # Save the datasets
    df.to_csv(save_dir / "all.csv")
    train.to_csv(splits_save_dir / "train.csv")
    test.to_csv(splits_save_dir / "test.csv")

    print("Data split successfully.")
    print("Train size:", len(train))
    print("Test size:", len(test))
    print(f"Data saved in {save_dir}")

    # Plot the dataset stats
    stats = get_dataset_stats(df)
    plot_dataset_stats(stats, save_dir)


# This script should only be used to split the data into training/validtion/testing
# sets at the start of the project or in case we want to separate the data in a different way.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Path to the single dataset CSV.")
    parser.add_argument("--balance", action='store_true', help="Balance the dataset.")
    parser.add_argument("--test_ratio", type=float, help="Validation ratio from the training set.", default=0.1)
    parser.add_argument("--seed")
    main(parser.parse_args())