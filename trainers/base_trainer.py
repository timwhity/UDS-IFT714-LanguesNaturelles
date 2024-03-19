from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, model, loss_fn, optimizer, scheduler, trainloader, validloader, testloader) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.trainloader = trainloader
        self.validloader = validloader
        self.testloader = testloader

    @abstractmethod
    def train(self, nb_training_steps: int):
        raise NotImplementedError
    
    @abstractmethod
    def validate(self):
        raise NotImplementedError
    
    @abstractmethod
    def test(self):
        raise NotImplementedError