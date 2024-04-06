import torch.nn as nn
from torch.nn.modules.module import T
import joblib
from sklearn.tree import DecisionTreeClassifier

class DecisionTreeUrl():

    def __init__(self, seed=None) -> None:
        super().__init__()
        self.tree = DecisionTreeClassifier(
            random_state=seed, criterion="gini", max_features="sqrt"
        )

    def forward(self, inputs, probs=True):
        if probs:
            return self.tree.predict_proba(inputs)[:, 1]
        else:
            return self.tree.predict(inputs)
    
    def train(self, inputs, targets):
        self.tree = self.tree.fit(inputs, targets)
    
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def load_state_dict(self, model_path):
        self.tree = joblib.load(model_path)