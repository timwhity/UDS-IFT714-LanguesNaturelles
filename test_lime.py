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
from utils.model_utils import load_model, load_config_data

URLS_TO_EXPLAIN = [
    # Benign
    "https://www.google.com",
    "https://jeremilevesque.com",
    "https://www.youtube.com/watch?v=j-t91Bzo8cE",
    "paypal.com.cgi.bin.webscr.cmd.flow.session.lohzumu98pjkwkwudgtj3ie6btlub.online775885d80a13c0db1f8e263663d3faee8d43b1bb6ca3ufquez.login.efzg5epaloginuk.uzmandoktorum.com/",
    
    # Malicious
    "www.angelfire.com/az2/svtool/index.html",
    "http://fywuw8ar.myutilitydomain.com/file/1fb4e1a4e2a2dadd0334d0dc641877d7/",
    "http://cheaproomsvalencia.com/Paypal/Support/ID-NUMB629/myaccount/signin"
]

def main(args):
    experiment_name = args.experiment_name
    splits_directory = Path(args.dataset_directory) / "splits"

    model_name, batch_size = load_config_data(experiment_name)

    max_seq_length = 512 # Load that from the dataset config
    device = ptu.get_device()

    # Load the model
    model, tokenizer, trainer_cls = load_model(model_name, experiment_name, device)
    loss_fn = torch.nn.BCELoss()

    trainer = trainer_cls(experiment_name,
                        model,
                        tokenizer,
                        loss_fn,
                        None,
                        None,
                        splits_directory,
                        batch_size,
                        device=device,
                        max_seq_length=max_seq_length,
                        limit=args.limit)

    save_dir = Path("explanations") / experiment_name

    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    explainer = UrlExplainer(save_dir=save_dir, class_names=["benign", "malicious"])
    explanations = explainer.explain_list(URLS_TO_EXPLAIN, trainer.predict)
    explainer.show_last_explanations()

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    add_default_arguments(parser)
    main(parser.parse_args())