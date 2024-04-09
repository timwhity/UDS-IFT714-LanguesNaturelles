import argparse

def add_default_arguments(parser: argparse.ArgumentParser):
    # Add default arguments to the parser
    parser.add_argument("experiment_name", type=str, help="The name of the experiment")
    parser.add_argument("dataset_directory", type=str, default="data/combined_dataset_12", help="The directory containing the dataset already splitted.")
    parser.add_argument("--num_workers", type=int, default=2, help="The number of workers to use for data loading")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of batches for training and testing. Used for CPU testing/training or debugging.")
    
    return parser

