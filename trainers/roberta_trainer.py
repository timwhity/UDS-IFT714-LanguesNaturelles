import torch
from tqdm import tqdm

from .base_trainer import BaseTrainer

class RobertaTrainer(BaseTrainer):
    def __init__(self, model, tokenizer, loss_fn, optimizer, scheduler, trainloader, validloader, testloader, classes, device, max_seq_length=2048) -> None:
        super().__init__(model, tokenizer, loss_fn, optimizer, scheduler, trainloader, validloader, testloader, classes, device)
        self.max_seq_length = max_seq_length
    
    def train(self, epoch_title: str = "Epoch"):
        self.model.train()

        total_loss = 0.0
        total_correct = 0

        for inputs, targets in tqdm(self.trainloader, desc=epoch_title):
            
            # Tokenize the inputs
            tokenized = self.tokenizer(inputs,
                                    max_length=self.max_seq_length,
                                    padding="max_length",
                                    truncation=True,
                                    return_attention_mask=True,
                                    add_special_tokens=True)
            inputs_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long, device=self.device)
            attention_masks = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs_ids, attention_masks=attention_masks)
            loss = self.loss_fn(outputs, targets)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            total_correct += (outputs == targets).sum().item()
        
        self.history["train"]["loss"].append(total_loss / len(self.trainloader))
        self.history["train"]["accuracy"].append(total_correct / len(self.trainloader.dataset))

        return self.history["train"]

    
    def validate(self, test: bool = False):
        hist_key = "valid"
        dataloader = self.validloader
        if test:
            hist_key = "test"
            dataloader = self.testloader

        total_loss = 0.0
        total_correct = 0

        self.model.eval()

        with torch.no_grad():
            for inputs, targets in dataloader:
                tokenized = self.tokenizer(inputs,
                                        max_length=self.max_seq_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True)
                inputs_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long, device=self.device)
                attention_masks = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
                targets = targets.to(self.device)


                outputs = self.model(inputs_ids, attention_masks=attention_masks)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                total_correct += (outputs == targets).sum().item()
        
        self.history[hist_key]["loss"].append(total_loss / len(self.validloader))
        self.history[hist_key]["accuracy"].append(total_correct / len(self.validloader.dataset))

        return self.history[hist_key]

    
    def test(self):
        return self.validate(test=True)