from base_trainer import BaseTrainer

class RobertaTrainer(BaseTrainer):
    def __init__(self, model, loss_fn, optimizer, scheduler, trainloader, validloader, testloader) -> None:
        super().__init__(model, loss_fn, optimizer, scheduler, trainloader, validloader, testloader)
    
    def train(self, nb_training_steps: int):
        pass
    
    def validate(self):
        pass
    
    def test(self):
        pass