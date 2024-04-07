import argparse
import torch
import transformers
from pathlib import Path

from models.roberta import RobertaUrl
from models.decision_tree import DecisionTreeUrl
from trainers.roberta_trainer import RobertaTrainer
from trainers.decision_tree_trainer import DecisionTreeTrainer
import utils.torch_utils as ptu
from data.data_utils import load_url_dataset
from trainers.trainer_metrics import TrainerMetrics
from utils.utils import add_default_arguments
from interpretability.url_explainer import UrlExplainer

splits_directory = Path("data/dataset_1/splits")

experiment_name = "efficient"
model_name, batch_size = load_config_data(experiment_name)

(_, _, testloader), classes = load_url_dataset(splits_directory, batch_size, num_workers=4, test=True)

max_seq_length = 512 # Load that from the dataset config
device = ptu.get_device()

# Load the model
model, tokenizer, trainer_cls = load_model(model_name, experiment_name)
model = model.to(device)
loss_fn = torch.nn.BCELoss()

trainer = trainer_cls(experiment_name,
                    model,
                    tokenizer,
                    loss_fn,
                    None,
                    None,
                    None,
                    None,
                    testloader,
                    classes,
                    device=device,
                    max_seq_length=512)

# Test prediction to be used for LIME
preds = trainer.predict(["br-icloud.com.br"])
print(preds)

explainer = LimeTextExplainer(class_names=["benign", "malicious"])
explanation = explainer.explain_instance("google.com", trainer.predict)

print("Explanation: ", explanation)
explanation.save_to_file("lime_google.html")