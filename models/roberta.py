import torch.nn as nn
from torch.nn.modules.module import T
from transformers import RobertaModel

class RobertaUrl(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        raw_output = self.roberta_model(input_ids, attention_mask, return_dict=True)
        pooler = raw_output["pooler_output"] # (B, 768)
        logit = self.fc(pooler) # TODO: Add attention head or concat the last hidden states?
        prob_malicious = self.sigmoid(logit)
        return prob_malicious

    # def _apply(self, fn):
    #     super(RobertaUrl, self)._apply(fn)
    #     self.roberta_model = fn(self.roberta_model)
    #     self.fc = fn(self.fc)
    #     self.sigmoid = fn(self.sigmoid)
    #     return self
