import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer

class RobertaTrainer(BaseTrainer):
    def __init__(self, experiment, model, tokenizer, loss_fn, optimizer, scheduler, trainloader, validloader, testloader, classes, device, max_seq_length=2048) -> None:
        super().__init__(experiment, "roberta", model, tokenizer, loss_fn, optimizer, scheduler, trainloader, validloader, testloader, classes, device)
        self.max_seq_length = max_seq_length

    def train(self, eval_each: int = 0, epoch_title: str = "Epoch"):
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        count_examples = 0
        count_batches = 0

        for batch_index, (inputs, targets) in enumerate(tqdm(self.trainloader, desc=epoch_title)):
            
            # Tokenize the inputs
            tokenized = self.tokenizer(inputs,
                                    max_length=self.max_seq_length,
                                    padding="max_length",
                                    truncation=True,
                                    return_attention_mask=True,
                                    add_special_tokens=True)
            inputs_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long, device=self.device)
            targets = targets.float().to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs_ids, attention_mask=attention_mask).squeeze(1)
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

        return self.metrics.get_metrics("train")
    
    def validate(self, test: bool = False):
        hist_key = "valid"
        dataloader = self.validloader
        metrics_update_fn = self.metrics.update_valid_metrics
        if test:
            hist_key = "test"
            dataloader = self.testloader
            metrics_update_fn = self.metrics.update_test_metrics

        total_loss = 0.0
        total_correct = 0
        count_examples = 0
        count_batches = 0

        self.model.eval()

        pbar = tqdm(range(len(dataloader)), desc=hist_key)

        with torch.no_grad():
            for inputs, targets in dataloader:
                tokenized = self.tokenizer(inputs,
                                        max_length=self.max_seq_length,
                                        padding="max_length",
                                        truncation=True,
                                        return_attention_mask=True,
                                        add_special_tokens=True)
                inputs_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long, device=self.device)
                attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long, device=self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs_ids, attention_mask=attention_mask).squeeze(1)
                preds = (outputs > 0.5).long()
                loss = self.loss_fn(outputs, targets.float())

                pbar.update()
                total_loss += loss.item()
                total_correct += (preds == targets).sum().item()
                count_examples += len(targets)
                count_batches += 1

        avg_loss = total_loss / count_batches
        accuracy = total_correct / count_examples
        metrics_update_fn(avg_loss, accuracy)

        return self.metrics.get_metrics(hist_key)
    
    def test(self):
        return self.validate(test=True)