import argparse
import torch
import transformers
import json

from models.roberta import RobertaUrl
from trainers.roberta_trainer import RobertaTrainer
import utils.torch_utils as ptu
from data.data_utils import load_url_dataset

# Load the dataset
num_workers = 2
batch_size = 16
num_epochs = 1
splits_directory = "data/splits"

(_, _, testloader), classes = load_url_dataset(splits_directory, batch_size, num_workers=num_workers, test=True)

nb_training_steps = len(testloader) * num_epochs
max_seq_length = 512
device = ptu.get_device()

# Load the model
model = RobertaUrl().to(device)
tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = transformers.get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=nb_training_steps)

model.load_state_dict(torch.load("models/trained/roberta_url.pth", map_location=device))

trainer = RobertaTrainer(model,
                         tokenizer,
                         loss_fn,
                         optimizer,
                         scheduler,
                         None,
                         None,
                         testloader,
                         classes,
                         device=device,
                         max_seq_length=max_seq_length)

test_hist = trainer.test(limit=7)
print(test_hist)

# Save test metrics to JSON file
with open("models/trained/test_metrics.json", "w") as f:
    json.dump(test_hist, f)


