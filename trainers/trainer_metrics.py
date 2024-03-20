import json
import matplotlib.pyplot as plt

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
                "accuracy": []
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

    def update_valid_metrics(self, loss, accuracy):
        self.metrics["valid"]["loss"].append(loss)
        self.metrics["valid"]["accuracy"].append(accuracy)
    
    def update_test_metrics(self, loss, accuracy):
        self.metrics["test"]["loss"].append(loss)
        self.metrics["test"]["accuracy"].append(accuracy)

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

    def save_metrics(self, filename):
        with open(filename, "w") as f:
            json.dump(self.metrics, f, indent=4)
    
    def random_metrics(self):
        import random
        for key in self.metrics:
            for subkey in self.metrics[key]:
                self.metrics[key][subkey] = [random.random() for _ in range(10)]
        return self.metrics
