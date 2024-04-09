import argparse
import torch
import transformers
import json

from models.roberta import RobertaUrl
from trainers.roberta_trainer import RobertaTrainer
from models.cnn import CNNUrl
from trainers.cnn_trainer import CNNTrainer
import utils.torch_utils as ptu
from utils.utils import add_default_arguments
from data.data_utils import load_url_dataset


def load_model(model_name: str):
    # Return the model, tokenizer, and trainer class for the given model name
    if model_name == "roberta":
        return RobertaUrl(), transformers.RobertaTokenizer.from_pretrained("roberta-base"), RobertaTrainer
    elif model_name == "cnn":
        return CNNUrl(), None, CNNTrainer
    else:
        raise ValueError(f"Invalid model name: {model_name}")

def main(args):
    # Load the dataset
    num_workers = args.num_workers
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    splits_directory = args.splits_directory

    experiment_name = args.experiment_name
    model_name = args.model_name

    (trainloader, validloader, testloader), classes = load_url_dataset(splits_directory, batch_size, num_workers=num_workers)

    nb_training_steps = len(trainloader) * num_epochs
    max_seq_length = 512 # Load that from the dataset config
    device = ptu.get_device()

    # Load the model
    model, tokenizer, trainer_cls = load_model(model_name)
    model.to(device)
    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = transformers.get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=nb_training_steps)

    trainer = trainer_cls(experiment_name,
                        model,
                        tokenizer,
                        loss_fn,
                        optimizer,
                        scheduler,
                        trainloader,
                        validloader,
                        testloader,
                        classes,
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
    parser.add_argument("--model_name", type=str, default="roberta", help="The name of the model to use", nargs="?", choices=["roberta", "cnn"])
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1, help="The number of epochs to train for")

    main(parser.parse_args())