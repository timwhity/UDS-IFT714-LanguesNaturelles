import torch.nn as nn

class MLPUrl(nn.Module):
    def __init__(self, input_size=512) -> None:
        super().__init__()
  
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size*70, 512),
            nn.ReLU(),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1)
        )
    
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        logits = self.model(x)
        probs = self.sigmoid(logits).squeeze(1)
        return probs
