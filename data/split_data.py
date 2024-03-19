import argparse
import pandas as pd
from pathlib import Path

filedir = Path(__file__).resolve().parent

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing data...", end="")
    # TODO: Preprocess the data if needed here
    print("Done.")
    return data

# MASSIVE TODO: KEEP THE PROPORTIONS OF CLASSES INTO EACH TRAIN/TEST SPLIT (DATASET ISN'T BALANCED)

def main(args):
    dataset_file = args.data
    test_ratio = args.test_ratio
    seed = args.seed

    df = pd.read_csv(dataset_file, index_col=0)
    df = preprocess_data(df)

    # Split the data into training, validation, and testing sets
    test = df.sample(frac=test_ratio, random_state=seed)
    train = df.drop(test.index)

    save_dir = filedir / "splits"
    if not save_dir.exists():
        save_dir.mkdir()

    # Save the datasets
    train.to_csv(save_dir / "train.csv")
    test.to_csv(save_dir / "test.csv")

    print("Data split successfully.")
    print("Train size:", len(train))
    print("Test size:", len(test))
    print(f"Data saved in {save_dir}")


# This script should only be used to split the data into training/validtion/testing
# sets at the start of the project or in case we want to separate the data in a different way.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str, help="Path to the single dataset CSV.")
    parser.add_argument("--test_ratio", type=float, help="Validation ratio from the training set.", default=0.1)
    parser.add_argument("--seed")
    main(parser.parse_args())