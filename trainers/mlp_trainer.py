import torch
import torch.nn as nn
import torch.optim as optim
from .base_trainer import BaseTrainer
import numpy as np
from typing import List

class MLPTrainer(BaseTrainer):
    def __init__(self, experiment, model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, limit=None, max_seq_length=512):
        super().__init__(experiment, "mlp", model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, limit)
        self.char_dic = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'/\\|_@#$%ˆ&*˜‘+-=<>()[]{}\n"
        self.nb_char_dic = 70
        assert len(self.char_dic) == self.nb_char_dic
        self.max_seq_length = max_seq_length

    def tokenize(self, url: str) -> np.array:
        """
        Input : string of the url
        Output : numpy array (nb_char_dic, max_seq_length)
        """
        sparse_vector = np.zeros(( self.nb_char_dic, self.max_seq_length))
        nb_accepted_characters = 0
        for c in url:
            if c in self.char_dic:
                sparse_vector[self.char_dic.index(c), nb_accepted_characters] = 1
                nb_accepted_characters += 1
            elif c.lower() in self.char_dic:
                sparse_vector[self.char_dic.index(c.lower()), nb_accepted_characters] = 1
                nb_accepted_characters += 1
            if nb_accepted_characters == self.max_seq_length:
                break
        return sparse_vector
	
    def tokenize_batch(self, urls: List[str]) -> np.array:
        """
        Input : list of urls
        Output : numpy array (len(urls), max_seq_length, nb_char_dic)
        """
        return np.array([self.tokenize(url) for url in urls])

    def train(self):
        # Training logic
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.trainloader):
            data = self.tokenize_batch(data)
            data, target = torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(target, dtype=torch.float32, device=self.device)
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
                data = self.tokenize_batch(data)
                data, target = torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(target, dtype=torch.float32, device=self.device)
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
                data = self.tokenize_batch(data)
                data, target = torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(target, dtype=torch.float32, device=self.device)
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
            data = self.tokenize_batch(texts)
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
            output = self.model(data.input_ids)
            probabilities = nn.functional.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
            return predicted_class, probabilities

