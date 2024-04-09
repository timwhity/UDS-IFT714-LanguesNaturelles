import torch
from tqdm import tqdm
from .base_trainer import BaseTrainer

class RobertaTrainer(BaseTrainer):
    def __init__(self, experiment, model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, limit = None, max_seq_length=2048) -> None:
        super().__init__(experiment, self._get_model_name(), model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, limit = None)
        self.max_seq_length = max_seq_length

    def _get_model_name(self):
        return "roberta"

    def predict(self, texts):

        # For LIME predictions, we need to train by batches
        def batch(iteratable, n=1):
            l = len(iteratable)
            for ndx in range(0, l, n):
                yield iteratable[ndx:min(ndx + n, l)]

        self.model.eval()
        batch_preds = []
        for text_batch in tqdm(batch(texts, n=32), total=len(texts)//32 + 1):
            tokenized = self.tokenizer(text_batch,
                                    max_length=self.max_seq_length,
                                    padding="max_length",
                                    truncation=True,
                                    return_attention_mask=True,
                                    add_special_tokens=True)
            inputs_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long, device=self.device)

            with torch.no_grad():
                outputs = self.model(inputs_ids, attention_mask=attention_mask).squeeze(1)
                
                prob_benign = 1 - outputs

                preds = torch.stack((prob_benign, outputs), dim=1)
                batch_preds.append(preds)

        batch_preds = torch.concat(batch_preds)

        return batch_preds.cpu().numpy()

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

            if accuracy > 0.995:
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
            for batch_index, (inputs, targets) in enumerate(dataloader):
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
    