import json
import matplotlib.pyplot as plt
import seaborn as sns

class TrainerMetrics(object):
    def __init__(self, config: dict = {}) -> None:
        # Each list contains lists for each epoch
        self.config = config
        self.metrics = self.clear_metrics()

    def clear_metrics(self):
        self.metrics = {
            "config": self.config,
            "train": {
                "loss": [],
                "accuracy": [],
                "lr": []
            },
            "valid": {
                "loss": [],
                "accuracy": []
            },
            "test": {
                "loss": [],
                "accuracy": [],
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0
            }
        }

        return self.metrics

    @staticmethod
    def from_file(filename, config_only=True):
        # Only stores the config into the metrics object
        with open(filename, "r") as f:
            metrics = json.load(f)

        trainer_metrics = TrainerMetrics(config=metrics["config"])

        if config_only:
            return trainer_metrics
        else:
            trainer_metrics.metrics = metrics
            return trainer_metrics
    
    def update_train_metrics(self, loss, accuracy, lr):
        self.metrics["train"]["loss"].append(loss)
        self.metrics["train"]["accuracy"].append(accuracy)
        self.metrics["train"]["lr"].append(lr)
        return self.metrics["train"]

    def update_valid_metrics(self, loss, accuracy):
        self.metrics["valid"]["loss"].append(loss)
        self.metrics["valid"]["accuracy"].append(accuracy)
        return self.metrics["valid"]
    
    def update_test_metrics(self, loss, accuracy, TP, TN, FP, FN):
        self.metrics["test"]["loss"].append(loss)
        self.metrics["test"]["accuracy"].append(accuracy)

        # For each batch, update the confusion matrix
        self.metrics["test"]["TP"] += TP 
        self.metrics["test"]["TN"] += TN
        self.metrics["test"]["FP"] += FP
        self.metrics["test"]["FN"] += FN

        # Get the TOTAL current confusion matrix values
        current_tp = self.metrics["test"]["TP"]
        current_fp = self.metrics["test"]["FP"]
        current_fn = self.metrics["test"]["FN"]

        # Calculate current precision/recall/f1
        precision = current_tp / (current_tp + current_fp) if (current_tp + current_fp) > 0 else 0
        recall = current_tp / (current_tp + current_fn) if (current_tp + current_fn) > 0 else 0

        # Update current precision/recall/f1 in the metrics
        self.metrics["test"]["precision"] = precision
        self.metrics["test"]["recall"] = recall
        self.metrics["test"]["f1"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return self.metrics["test"]

    def get_metrics(self, key=None):
        if key:
            return self.metrics[key]
        else:
            return self.metrics
        
    def plot_metrics(self, filename=None):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Metrics")

        # Training
        axs[0, 0].plot(self.metrics["train"]["loss"], label="Train Loss")
        axs[0, 0].set_title("Train Loss")
        axs[0, 1].plot(self.metrics["train"]["accuracy"], label="Train Accuracy")
        axs[0, 1].set_title("Train Accuracy")
        axs[0, 2].plot(self.metrics["train"]["lr"], label="Learning Rate")
        axs[0, 2].set_title("Learning Rate")
        
        # Validation
        axs[1, 0].plot(self.metrics["valid"]["loss"], label="Valid Loss")
        axs[1, 0].set_title("Valid Loss")
        axs[1, 1].plot(self.metrics["valid"]["accuracy"], label="Valid Accuracy")
        axs[1, 1].set_title("Valid Accuracy")
        axs[1, 2].remove()

        plt.tight_layout()
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

        # Plot confusion matrix
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        sns.heatmap([[self.metrics["test"]["TP"], self.metrics["test"]["FP"]], 
                     [self.metrics["test"]["FN"], self.metrics["test"]["TN"]]], annot=True, fmt="d", ax=ax)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Prediction")
        ax.set_xticklabels(["Malicious", "Benign"])
        ax.set_yticklabels(["Malicious", "Benign"])

        plt.tight_layout()
        if filename:
            plt.savefig(filename.parent / (filename.stem + "_confusion_matrix.png"))
        else:
            plt.show()

    def save_metrics(self, filename):
        with open(filename, "w") as f:
            json.dump(self.metrics, f, indent=4)
    
    def random_metrics(self):
        import random
        for key in self.metrics:
            for subkey in self.metrics[key]:
                self.metrics[key][subkey] = [random.random() for _ in range(10)]
        return self.metrics
