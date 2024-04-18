import argparse
import torch
import transformers
from pathlib import Path

import utils.torch_utils as ptu
from utils.utils import add_default_arguments
from data.data_utils import load_url_dataset
from model_utils import load_model
from trainers.decision_tree_trainer import DecisionTreeTrainer

def main(args):
    # Load the dataset
    num_workers = args.num_workers
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    splits_directory = Path(args.dataset_directory) / "splits"

    experiment_name = args.experiment_name
    model_name = args.model_name

    nb_training_steps = 27597
    max_seq_length = 512 # Load that from the dataset config
    device = ptu.get_device()

    # Load the model
    model, tokenizer, trainer_cls = load_model(model_name, experiment_name, device)
    
    if trainer_cls is DecisionTreeTrainer:
        # DecisionTreeTrainer does not require a tokenizer
        loss_fn = None
        optimizer = None
        scheduler = None
    else:
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        scheduler = transformers.get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=nb_training_steps)

    trainer = trainer_cls(experiment_name,
                        model,
                        tokenizer,
                        loss_fn,
                        optimizer,
                        scheduler,
                        splits_directory,
                        batch_size,
                        device=device,
                        max_seq_length=max_seq_length,
                        limit=args.limit)


    for epoch in range(num_epochs):
        train_metrics = trainer.train()

    trainer.save_model()
    trainer.save_experiment_metrics() # Save preleminary metrics

    test_hist = trainer.test()
    print(test_hist)

    trainer.save_experiment_metrics() # Save metrics after testing

if '__main__' == __name__:
    parser = argparse.ArgumentParser(description="Train a model on the URL dataset")
    parser = add_default_arguments(parser)
    parser.add_argument("--model_name", type=str, default="roberta", help="The name of the model to use", nargs="?", choices=["roberta", "decision_tree", "cnn", "bert", "mlp"])
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1, help="The number of epochs to train for")

    main(parser.parse_args())