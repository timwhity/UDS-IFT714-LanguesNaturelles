import argparse
import torch
import transformers
from pathlib import Path

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

def main(args):
    # Load the dataset
    num_workers = args.num_workers
    splits_directory = args.splits_directory

    experiment_name = args.experiment_name
    model_name, batch_size = load_config_data(experiment_name)

    (_, _, testloader), classes = load_url_dataset(splits_directory, batch_size, num_workers=num_workers, test=True)

    max_seq_length = 512 # Load that from the dataset config
    device = ptu.get_device()

    # Load the model
    model, tokenizer, trainer_cls = load_model(model_name, experiment_name)
    model = model.to(device)
    loss_fn = torch.nn.BCELoss()

    trainer = trainer_cls(experiment_name,
                        model,
                        tokenizer,
                        loss_fn,
                        None,
                        None,
                        None,
                        None,
                        testloader,
                        classes,
                        device=device,
                        max_seq_length=max_seq_length,
                        limit=args.limit)

    test_hist = trainer.test()
    print(test_hist)

    trainer.save_experiment_metrics(prefix="test") # Save metrics after testing

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="Train a model on the URL dataset")
    parser = add_default_arguments(parser)

    main(parser.parse_args())