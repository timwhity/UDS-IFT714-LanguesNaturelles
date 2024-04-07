from .base_trainer import BaseTrainer
from data.feature_extractor import FeatureExtractor
from tqdm import tqdm

class DecisionTreeTrainer(BaseTrainer):
    def __init__(self, experiment, model, tokenizer, loss_fn, optimizer, scheduler, trainloader, validloader, testloader, classes, device, limit=None, max_seq_length=2048) -> None:
        super().__init__(experiment, "decision_tree", model, tokenizer, loss_fn, optimizer, scheduler, trainloader, validloader, testloader, classes, device, limit)
        self.feature_extractor = FeatureExtractor()

    def predict(self, texts):
        probs = self.model(self.feature_extractor.extract_batch(texts), probs=True)
        return probs

    def train(self, eval_each: int = 0, epoch_title: str = "Epoch"):
        X_train, y_train = self.trainloader.get_all_data()

        # Train the model (in one shot)
        pbar = tqdm(total=1, desc=epoch_title)
        X_train = self.feature_extractor.extract_features(X_train)
        self.model.train(X_train, y_train)
        pbar.update(1)
        pbar.close()

        # Calculate the metrics
        train_accuracy = (y_train == self.model(X_train)).mean()
        self.metrics.update_train_metrics(0.0, train_accuracy, 0.0) # Loss is not calculated for decision trees 

        self.validate()

        return self.metrics.get_metrics("train")

    def validate(self, test: bool = False):
        X_valid, y_valid = self.testloader.get_all_data() if test else self.validloader.get_all_data()

        valid_accuracy = (y_valid == self.model(self.feature_extractor.extract_features(X_valid))).mean()
        self.metrics.update_valid_metrics(0.0, valid_accuracy, 0.0)


    def test(self):
        return self.validate(test=True)