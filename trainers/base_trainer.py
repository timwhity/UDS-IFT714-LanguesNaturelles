import torch
from abc import ABC, abstractmethod
from pathlib import Path
import warnings
from .trainer_metrics import TrainerMetrics

class BaseTrainer(ABC):
    def __init__(self, experiment, model_name, model, tokenizer, loss_fn, optimizer, scheduler, trainloader, validloader, testloader, classes, device) -> None:
        self.experiment_name = experiment
        self.experiment_dir = Path("models/trained") / self.experiment_name

        if self.experiment_dir.exists():
            warnings.warn(f"Experiment {self.experiment_name} already exists. Files might be overwritten.")
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.model_name = model_name
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader
        self.classes = classes

        list_classes = list(map(lambda x: int(x), self.classes)) # JSON doesn't support numpy's types

        config_info = {
            "model": str(self.model_name),
            "batch_size": int(self.testloader.batch_size),
            # "tokenizer": str(self.tokenizer),
            "loss_fn": str(self.loss_fn),
            "optimizer": str(self.optimizer),
            "scheduler": str(self.scheduler),
            "trainloader": str(self.trainloader), # Maybe include here some stats about the dataset? (e.g. balanced, imbalanced, etc.)
            "validloader": str(self.validloader), # Maybe include here some stats about the dataset? (e.g. balanced, imbalanced, etc.)
            "testloader": str(self.testloader),   # Maybe include here some stats about the dataset? (e.g. balanced, imbalanced, etc.)
            "classes": list_classes
        }
        self.metrics = TrainerMetrics(config=config_info)
        

    def reset_metrics(self):
        self.metrics = TrainerMetrics()

    def save_model(self):
        # Save the model
        torch.save(self.model.state_dict(), self.experiment_dir / "roberta_url.pth")

    def save_experiment_metrics(self, prefix=None):
        prefix_str = ""
        if prefix:
            prefix_str = f"{prefix}_"

        # Save the metrics
        self.metrics.save_metrics(self.experiment_dir / (prefix_str + "metrics.json"))

        # Plot and save the metrics
        self.metrics.plot_metrics(self.experiment_dir / (prefix_str + "metrics.png"))

        print(f"Experiment (model, metrics & config) saved to {self.experiment_dir}")

    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def validate(self):
        raise NotImplementedError
    
    @abstractmethod
    def test(self):
        raise NotImplementedError