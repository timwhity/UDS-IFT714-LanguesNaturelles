import argparse
import torch

from data.data_utils import load_url_dataset

def main(args):
    num_workers = args.num_workers
    batch_size = args.batch_size
    splits_directory = args.splits_directory

    (trainloader, validloader, testloader), classes = load_url_dataset(splits_directory, batch_size, num_workers=num_workers)

    print("Hello World")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a simple program")
    parser.add_argument("--splits_directory", type=str, default="data/splits", help="The directory containing the train, valid, and test splits")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="The ratio of the training set, the rest is the validation set.")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size for the dataloaders")
    parser.add_argument("--num_workers", type=int, default=4, help="The number of workers for the dataloaders")
    pargs = parser.parse_args()
    main(pargs)