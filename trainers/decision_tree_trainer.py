from .base_trainer import BaseTrainer
from data.feature_extractor import FeatureExtractor
from tqdm import tqdm
from data.data_utils import load_full_url_dataset

class DecisionTreeTrainer(BaseTrainer):
    def __init__(self, experiment, model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, max_seq_length: int = 256, limit = None) -> None:
        super().__init__(experiment, "decision_tree", model, tokenizer, loss_fn, optimizer, scheduler, splits_directory, batch_size, device, limit = None)
        self.feature_extractor = FeatureExtractor()
        self.trainset, self.validset, self.testset, self.classes = load_full_url_dataset(splits_directory, batch_size, num_workers=4)

    def predict(self, texts):
        probs = self.model(self.feature_extractor.extract_batch(texts), probs=True)
        return probs

    def train(self, eval_each: int = 0, epoch_title: str = "Epoch"):
        X_train, y_train = self.trainset[:]

        # Train the model (in one shot)
        pbar = tqdm(total=1, desc=epoch_title)
        X_train = self.feature_extractor.extract_batch(X_train)
        self.model.train(X_train, y_train)
        pbar.update(1)
        pbar.close()

        # Calculate the metrics
        train_accuracy = (y_train == self.model(X_train)).mean()
        self.metrics.update_train_metrics(0.0, train_accuracy, 0.0) # Loss is not calculated for decision trees 

        self.validate()
        self.test()

        return self.metrics.get_metrics("train")

    def validate(self, test: bool = False):
        X_valid, y_valid = self.testset[:] if test else self.validset[:]

        preds = self.model(self.feature_extractor.extract_batch(X_valid))
        acc = (y_valid == preds).mean()
        
        if test:
            TP = ((preds == 1) & (y_valid == 1)).sum().item()
            TN = ((preds == 0) & (y_valid == 0)).sum().item()
            FP = ((preds == 1) & (y_valid == 0)).sum().item()
            FN = ((preds == 0) & (y_valid == 1)).sum().item()
            self.metrics.update_test_metrics(0.0, acc, TP, TN, FP, FN)
        else:
            self.metrics.update_valid_metrics(0.0, acc)

    def test(self):
        return self.validate(test=True)
    
    def save_model(self):
        self.model.save_state_dict(self.experiment_dir / "decision_tree_url.pkl")