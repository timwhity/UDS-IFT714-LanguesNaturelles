import torch
import torch.nn as nn
import torch.optim as optim
from base_trainer import BaseTrainer

class MLPTrainer(BaseTrainer):
    def __init__(self, experiment, model_name, model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, input_size, hidden_size, output_size, limit=None):
        super().__init__(experiment, model_name, model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, limit)
        
        # Define the architecture of the MLP
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(device)

    def train(self):
        # Training logic
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

    def validate(self):
        # Validation logic
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.validloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.validloader.dataset)
        accuracy = 100. * correct / len(self.validloader.dataset)
        return val_loss, accuracy

    def test(self):
        # Test logic
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.testloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.testloader.dataset)
        accuracy = 100. * correct / len(self.testloader.dataset)
        return test_loss, accuracy

    def predict(self, texts):
        # Prediction logic
        self.model.eval()
        with torch.no_grad():
            data = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(self.device)
            output = self.model(data.input_ids)
            probabilities = nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            return predicted_class, probabilities

