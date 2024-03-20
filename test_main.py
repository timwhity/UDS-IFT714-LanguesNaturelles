import argparse
import torch
import transformers
import json
from pathlib import Path

from models.roberta import RobertaUrl
from trainers.roberta_trainer import RobertaTrainer
import utils.torch_utils as ptu
from data.data_utils import load_url_dataset
from trainers.trainer_metrics import TrainerMetrics

def load_model(model_name: str):
    # Return the model, tokenizer, and trainer class for the given model name
    if model_name == "roberta":
        return RobertaUrl(), transformers.RobertaTokenizer.from_pretrained("roberta-base"), RobertaTrainer
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
def load_config_data(experiment_name: str):
    # Load the config from the experiment directory
    experiment_dir = Path("models/trained") / experiment_name
    metrics = TrainerMetrics.from_file(experiment_dir / "metrics.json", config_only=True)
    
    model_name = metrics.config["model"]
    batch_size = metrics.config["batch_size"]

    return model_name, batch_size

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
    model, tokenizer, trainer_cls = load_model(model_name)
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
                        max_seq_length=max_seq_length)

    test_hist = trainer.test()
    print(test_hist)

    trainer.save_experiment_metrics(prefix="test") # Save metrics after testing

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="Train a model on the URL dataset")
    parser.add_argument("experiment_name", type=str, help="The name of the experiment")
    parser.add_argument("--num_workers", type=int, default=2, help="The number of workers to use for data loading")
    parser.add_argument("--splits_directory", type=str, default="data/splits", help="The directory containing the dataset splits")

    main(parser.parse_args())