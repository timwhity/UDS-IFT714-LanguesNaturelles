import torch.nn as nn

class MLPUrl(nn.Module):
    def __init__(self, input_size=512) -> None:
        super().__init__()
  
		# Define the architecture of the MLP
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )
    
        self.sigmoid = nn.Sigmoid()
  
    def forward(self, x):
        logits = self.model(x)
        probs = self.sigmoid(logits)
        return probs
