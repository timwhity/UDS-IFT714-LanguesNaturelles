import torch.nn as nn


class CNNUrl(nn.Module):

	def __init__(self) -> None:
		super().__init__()

		# Embedding : (max_seq_length=256, nb_char_dic=70) -> (128, 16)	???
		# Conv + ReLu : (128, 16) -> (64, 1, 8)
		# Maxpooling : (64, 1, 8) -> (32, 1, 8)
		# Conv + ReLu : (32, 1, 8) -> (16, 1, 16)
		# Maxpooling : (16, 1, 16) -> (8, 1, 16)
		# Conv + ReLu : (8, 1, 16) -> (8, 1, 32)
		# Maxpooling : (8, 1, 32) -> (1, 1, 32)
		# Linear + ReLu : (1, 1, 32) -> (32)
		# Linear + sigmoid : (32) -> (1)

		self.nn = nn.Sequential(
			nn.Conv1d(128, 64, 8),
			nn.ReLU(),
			nn.MaxPool1d(2),
			nn.Conv1d(64, 32, 16),
			nn.ReLU(),
			nn.MaxPool1d(2),
			nn.Conv1d(32, 8, 32),
			nn.ReLU(),
			nn.MaxPool1d(8),
			nn.Flatten(),
			nn.Linear(32, 1),
			nn.Sigmoid()
		)
  

	def forward(self, x):
		embeded = x
		return self.nn(embeded)

		