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

    def forward(self, inputs, probs=False):
        if probs:
            proba = self.tree.predict_proba(inputs)
            return proba
        else:
            return self.tree.predict(inputs)
    
    def train(self, inputs, targets):
        self.tree = self.tree.fit(inputs, targets)
    
    def __call__(self, inputs, probs=False):
        return self.forward(inputs, probs)
    
    def load_state_dict(self, model_path):
        self.tree = joblib.load(model_path)[0]
