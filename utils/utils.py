import argparse

def add_default_arguments(parser: argparse.ArgumentParser):
    # Add default arguments to the parser
    parser.add_argument("experiment_name", type=str, help="The name of the experiment")
    parser.add_argument("--num_workers", type=int, default=2, help="The number of workers to use for data loading")
    parser.add_argument("--splits_directory", type=str, default="data/splits", help="The directory containing the dataset splits")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of batches for training and testing. Used for CPU testing/training or debugging.")
    
    return parser
