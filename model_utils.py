import torch
import transformers
from pathlib import Path

from models.roberta import RobertaUrl
from models.decision_tree import DecisionTreeUrl

from trainers.trainer_metrics import TrainerMetrics
from trainers.roberta_trainer import RobertaTrainer
from trainers.decision_tree_trainer import DecisionTreeTrainer

def load_model(model_name: str, experiment_name: str, device):
    experiment_dir = Path("models/trained") / experiment_name
    # Return the model, tokenizer, and trainer class for the given model name
    if model_name == "roberta":
        model = RobertaUrl()

        to_load = experiment_dir / "roberta_url.pth"
        if to_load.exists():
            model.load_state_dict(torch.load(to_load))

        return model.to(device), transformers.RobertaTokenizer.from_pretrained("roberta-base"), RobertaTrainer
    elif model_name == "decision_tree":
        model = DecisionTreeUrl()

        to_load = experiment_dir / "decision_tree_url.pkl"
        if to_load.exists():
            model.load_state_dict(to_load)
        return model, None, DecisionTreeTrainer
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
def load_config_data(experiment_name: str):
    # Load the config from the experiment directory
    experiment_dir = Path("models/trained") / experiment_name
    metrics = TrainerMetrics.from_file(experiment_dir / "metrics.json", config_only=True)
    
    model_name = metrics.config["model"]
    batch_size = metrics.config["batch_size"]

    return model_name, batch_size
