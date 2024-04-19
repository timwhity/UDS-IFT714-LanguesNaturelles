import torch.nn as nn


class CNNUrl(nn.Module):

	def __init__(self) -> None:
		super().__init__()
  
		################## ARCHITECTURE ##################
		# Entrée : (nb_char_dic=70 features, max_seq_length=256)
		# 0) Flatten : (70, 256) -> (70*256)
		# 1) Linear + ReLu : (70*256) -> (4*256)
		# 1 bis) Reshape : (4*256) -> (4, 256)
		# 2) Conv 1d + ReLu (8 filtres de taille 5 avec padding) : (4, 256) -> (8, 256)
		# 3) Maxpooling : (8, 256) -> (8, 128)
		# 4) Conv 1d + ReLu (16 filtres de taille 5 avec padding) : (8, 128) -> (16, 128)
		# 5) Maxpooling : (16, 128) -> (16, 64)
		# 6) Conv 1d + ReLu (32 filtres de taille 5 avec padding) : (16, 64) -> (32, 64)
		# 7) Maxpooling : (32, 64) -> (32, 32)
		# 8) Conv 1d + ReLu (64 filtres de taille 5 avec padding) : (32, 32) -> (64, 32)
		# 9) Maxpooling : (64, 32) -> (64, 16)
		# 10) Conv 1d + ReLu (128 filtres de taille 5 avec padding) : (64, 16) -> (128, 16)
		# 11) Maxpooling : (128, 16) -> (128, 8)
		# 12) Flatten : (128, 8) -> (1024)
		# 13) Linear + ReLu : (1024) -> (64)
		# 14) Linear + sigmoid : (64) -> (1)
  
		self.nn2 = nn.Sequential(
			nn.Flatten(),
			nn.Linear(70*256, 4*256),
			nn.ReLU(),
			nn.Unflatten(1, (4, 256)),
			nn.Conv1d(4, 8, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(8, 16, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(16, 32, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(32, 64, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(64, 128, kernel_size=5, padding=2),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Flatten(),
			nn.Linear(128*8, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
		)
  
	def forward(self, x):
		return self.nn2(x)
