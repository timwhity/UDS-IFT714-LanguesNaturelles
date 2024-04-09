import torch
from tqdm import tqdm
from .roberta_trainer import RobertaTrainer

class BertTrainer(RobertaTrainer):
    def __init__(self, experiment, model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, limit = None, max_seq_length=2048) -> None:
        super().__init__(experiment, model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, limit=limit, max_seq_length=max_seq_length)

    def _get_model_name(self):
        return "bert"

    def predict(self, texts):
        return super().predict(texts)

    def train(self, eval_each: int = 0, epoch_title: str = "Epoch"):
        return super().train(eval_each, epoch_title)
    
    def validate(self, test: bool = False):
        return super().validate(test)
    
    def test(self):
        return super().test()
    