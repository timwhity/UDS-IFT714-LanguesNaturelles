import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("darkgrid")

def get_dataset_stats(dataset):
    return {
        "num_samples": len(dataset),
        "num_benign": len(dataset[dataset["result"] == 0]),
        "num_malicious": len(dataset[dataset["result"] == 1]),
    }

def plot_dataset_stats(stats, save_dir=None):
    
    # Plot a pie chart of the number of benign and malicious samples
    fig, ax = plt.subplots()
    ax.pie([stats["num_benign"], stats["num_malicious"]], labels=["Benign ({})".format(stats["num_benign"]), "Malicious ({})".format(stats["num_malicious"])], autopct="%1.1f%%")
    ax.set_title(f"Dataset Stats\nNum Samples: {stats['num_samples']}")


    if save_dir is not None:
        save_dir = Path(save_dir) / "stats.png"
        plt.savefig(save_dir)
    else:
        plt.show()


def main(args):

    # Load the dataset
    dataset = pd.read_csv(args.dataset)
    stats = get_dataset_stats(dataset)
    plot_dataset_stats(stats, args.save_dir)


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument("dataset", type=str, help="The dataset to train on.")
    parser.add_argument("--save_dir", type=str, help="The directory to save the trained model.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed to use.") 
    main(parser.parse_args())
