import argparse
import torch
import transformers
from pathlib import Path

from models.roberta import RobertaUrl
from trainers.roberta_trainer import RobertaTrainer
import utils.torch_utils as ptu
from data.data_utils import load_url_dataset
from trainers.trainer_metrics import TrainerMetrics
from utils.utils import add_default_arguments
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt


def load_model(model_name: str, experiment_name: str):
    experiement_dir = Path("models/trained") / experiment_name
    # Return the model, tokenizer, and trainer class for the given model name
    if model_name == "roberta":
        model = RobertaUrl()
        model.load_state_dict(torch.load(experiement_dir / "roberta_url.pth"))
        return model, transformers.RobertaTokenizer.from_pretrained("roberta-base"), RobertaTrainer
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
def load_config_data(experiment_name: str):
    # Load the config from the experiment directory
    experiment_dir = Path("models/trained") / experiment_name
    metrics = TrainerMetrics.from_file(experiment_dir / "metrics.json", config_only=True)
    
    model_name = metrics.config["model"]
    batch_size = metrics.config["batch_size"]

    return model_name, batch_size


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