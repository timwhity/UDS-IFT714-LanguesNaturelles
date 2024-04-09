import torch
import torch.nn as nn
from transformers import BertModel

class BertUrl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        raw_output = self.bert_model(input_ids, attention_mask, return_dict=True)
        pooler = raw_output["pooler_output"] # (B, 768)
        logit = self.fc(pooler)
        prob_malicious = self.sigmoid(logit)
        return prob_malicious
        
