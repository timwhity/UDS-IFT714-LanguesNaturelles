import argparse
import pandas as pd
from pathlib import Path

def get_dataset_stats(dataset):
    len_moy = 0
    len_max = 0
    len_min = 1000000
    nb_benign = 0
    nb_malicious = 0
    for i in range(len(dataset)):
        len_moy += len(dataset["url"][i])
        len_max = max(len_max, len(dataset["url"][i]))
        len_min = min(len_min, len(dataset["url"][i]))
        if dataset["result"][i] == 0:
            nb_benign += 1
        else:
            nb_malicious += 1
    len_moy /= len(dataset)
    return {
        "Nombre total d'url": len(dataset),
        "Nombre d'url bénines": nb_benign,
        "Nombres d'url malicieuses": nb_malicious,
        "Longueur moyenne": len_moy,
        "Longueur maximum": len_max,
        "Longueur mininimum": len_min
    }

def save_dataset_stats(stats, save_dir, latex=False):
    text = ""
    if latex:
        text += "\\begin{table}[]\n"
        text += "\\begin{tabular}{|l|l|}\n"
        text += "\\hline\n"
        text += "Statistique & Valeur \\\\ \n"
        text += "\\hline\n"
        for key, value in stats.items():
            text += f"{key} & {value} \\\\ \n"
        text += "\\hline\n"
        text += "\\end{tabular}\n"
        text += "\\end{table}\n"
    else:
        for key, value in stats.items():
            text += f"{key}: {value}\n"

    if save_dir is not None:
        save_dir = Path(save_dir) / "stats.txt"
        with open(save_dir, "w") as f:
            f.write(text)
    else:
        print(text)


def main(args):

    # Load the dataset
    dataset = pd.read_csv(args.dataset)
    stats = get_dataset_stats(dataset)
    save_dataset_stats(stats, args.save_dir, args.latex)


if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="Train a model on a dataset.")
    parser.add_argument("dataset", type=str, help="The dataset to train on.")
    parser.add_argument("--save_dir", type=str, help="The directory to save the trained model.")
    parser.add_argument("--latex", action="store_true", help="Whether to save the stats in a latex table format.")
    parser.add_argument("--seed", type=int, default=42, help="The random seed to use.") 
    main(parser.parse_args())
