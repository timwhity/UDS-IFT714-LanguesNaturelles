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
            data = self.tokenizer.tokenize(text_batch)
            data, target = torch.tensor(data, dtype=torch.float32, device=self.device), torch.tensor(target, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                prob_malicious = self.model(data)
                prob_benign = 1 - prob_malicious

                preds = torch.stack((prob_benign, prob_malicious), dim=1)
                batch_preds.append(preds)

        batch_preds = torch.cat(batch_preds)
        return batch_preds.cpu().numpy()
    

    def train(self, eval_each: int = 0, epoch_title: str = "Epoch"):
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        count_examples = 0
        count_batches = 0

        for batch_index, (data, targets) in enumerate(tqdm(self.trainloader, desc=epoch_title)):
            data = self.tokenizer.tokenize(data)
            data = torch.tensor(data, dtype=torch.float32, device=self.device)
            targets = targets.float().to(self.device)
            self.optimizer.zero_grad()

            outputs = self.model(data)
            preds = (outputs > 0.5).long()

            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_correct += (preds == targets).sum().item()

            # Keep the count of examples and batches so the metrics are accurate.
            count_examples += len(targets)
            count_batches += 1

            lr = self.optimizer.param_groups[0]["lr"]
            avg_loss = total_loss / count_batches
            accuracy = total_correct / count_examples
            self.metrics.update_train_metrics(avg_loss, accuracy, lr)

            if (eval_each > 0) and (batch_index % eval_each == 0):
                self.validate()

            if batch_index > (len(self.trainloader) // 10) and accuracy > 0.995:
                break

            if self.limit and (batch_index >= self.limit): # Break prematurely for debugging on CPU or poor GPU
                break

        self.test()

        return self.metrics.get_metrics("train")

    def validate(self, test: bool = False):
        desc = "valid"
        dataloader = self.validloader
        if test:
            desc = "test"
            dataloader = self.testloader
            TP = 0 # Confusion matrix values counters
            TN = 0
            FP = 0
            FN = 0

        total_loss = 0.0
        total_correct = 0
        count_examples = 0
        count_batches = 0

        self.model.eval()

        pbar = tqdm(range(len(dataloader)), desc=desc)

        with torch.no_grad():
            for batch_index, (data, targets) in enumerate(dataloader):
                data = self.tokenizer.tokenize(data)
                data = torch.tensor(data, dtype=torch.float32, device=self.device)
                targets = targets.float().to(self.device)
                
                outputs = self.model(data)
                preds = (outputs > 0.5).long()
                loss = self.loss_fn(outputs, targets)

                if test:
                    TP += ((preds == 1) & (targets == 1)).sum().item()
                    TN += ((preds == 0) & (targets == 0)).sum().item()
                    FP += ((preds == 1) & (targets == 0)).sum().item()
                    FN += ((preds == 0) & (targets == 1)).sum().item()

                pbar.update()
                total_loss += loss.item()
                total_correct += (preds == targets).sum().item()
                count_examples += len(targets)
                count_batches += 1

                if self.limit and (batch_index >= self.limit): # Break prematurely for debugging on CPU or poor GPU
                    break

        avg_loss = total_loss / count_batches
        accuracy = total_correct / count_examples

        if test:
            return self.metrics.update_test_metrics(avg_loss, accuracy, TP, TN, FP, FN)
        else:
            return self.metrics.update_valid_metrics(avg_loss, accuracy)

    def test(self):
        return self.validate(test=True)


