from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model, tokenizer, loss_fn, optimizer, scheduler, trainloader, validloader, testloader, classes, device) -> None:
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

        self.history = {
            "train": {
                "loss": [],
                "accuracy": [],
                "lr": []
            },
            "valid": {
                "loss": [],
                "accuracy": []
            },
            "test": {
                "loss": [],
                "accuracy": []
            }
        }

    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def validate(self):
        raise NotImplementedError
    
    @abstractmethod
    def test(self):
        raise NotImplementedError