import torch
import torch.nn as nn
import torch.optim as optim
from .base_trainer import BaseTrainer
import numpy as np
from typing import List
from torchinfo import summary
from tqdm import tqdm

class MLPTrainer(BaseTrainer):
    def __init__(self, experiment, model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, limit=None, max_seq_length=512):
        super().__init__(experiment, "mlp", model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, limit)
    
    def predict(self, texts):
        self.model.eval()

        batch_preds = []
        for text_batch in tqdm(self.batch(texts, n=32), total=len(texts)//32 + 1):
            tokenized = self.tokenizer.tokenize(text_batch)
            data, target = torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(target, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                prob_malicious = self.model(data)
                prob_benign = 1 - prob_malicious

                preds = torch.stack((prob_benign, prob_malicious), dim=1)
                batch_preds.append(preds)

        batch_preds = torch.cat(batch_preds)
        return batch_preds.cpu().numpy()
    

    def train(self, eval_each: int = 0, epoch_title: str = "Epoch"):
        # Training logic
        self.model.train()
        for batch_idx, (data, target) in enumerate(tqdm(self.trainloader, desc=epoch_title)):
            data = self.tokenizer.tokenize(data)
            data, target = torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(target, dtype=torch.float32, device=self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

    def validate(self, test: bool = False):
        # Validation logic
        self.model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.validloader:
                data = self.tokenizer.tokenize(data)
                data, target = torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(target, dtype=torch.float32, device=self.device)
                output = self.model(data)
                val_loss += self.loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        val_loss /= len(self.validloader.dataset)
        accuracy = 100. * correct / len(self.validloader.dataset)
        return val_loss, accuracy

    def test(self):
        return self.validate(test=True)


