import argparse
import torch
import transformers
from pathlib import Path
import matplotlib.pyplot as plt

from models.roberta import RobertaUrl
from models.decision_tree import DecisionTreeUrl
from trainers.roberta_trainer import RobertaTrainer
from trainers.decision_tree_trainer import DecisionTreeTrainer
from data.feature_extractor import FeatureExtractor
import utils.torch_utils as ptu
from data.data_utils import load_url_dataset
from trainers.trainer_metrics import TrainerMetrics
from utils.utils import add_default_arguments
from model_utils import load_config_data, load_model
from sklearn import tree

def main(args):
    # Load the dataset
    num_workers = args.num_workers
    splits_directory = Path(args.dataset_directory) / "splits"

    experiment_name = args.experiment_name
    model_name, batch_size = load_config_data(experiment_name)

    max_seq_length = 512 # Load that from the dataset config
    device = ptu.get_device()

    # Load the model
    model, tokenizer, trainer_cls = load_model(model_name, experiment_name, device)
    loss_fn = torch.nn.BCELoss()

    trainer = trainer_cls(experiment_name,
                        model,
                        tokenizer,
                        loss_fn,
                        None,
                        None,
                        splits_directory,
                        batch_size,
                        device=device,
                        max_seq_length=max_seq_length,
                        limit=args.limit)

    test_hist = trainer.validate()
    print(test_hist)

    print(trainer.model.tree.tree_.max_depth)
    tree.plot_tree(trainer.model.tree, max_depth=8, class_names=["benign", "malicious"])
    plt.show()

    #trainer.save_experiment_metrics(prefix="valid") # Save metrics after testing

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="Train a model on the URL dataset")
    parser = add_default_arguments(parser)

    main(parser.parse_args())