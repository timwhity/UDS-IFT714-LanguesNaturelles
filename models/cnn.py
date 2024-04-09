import torch.nn as nn


class CNNUrl(nn.Module):

	def __init__(self) -> None:
		super().__init__()

		# Entrée : (nb_char_dic=70 features, max_seq_length=256)
		# 1) Conv 1d + ReLu (30 filtres de taille 3 avec padding) : (70, 256) -> (30, 256)
		# 2) Maxpooling : (30, 256) -> (30, 128)
		# 3) Conv 1d + ReLu (40 filtres de taille 3 avec padding) : (30, 128) -> (40, 128)
		# 4) Maxpooling : (40, 128) -> (40, 64)
		# 5) Conv 1d + ReLu (50 filtres de taille 3 avec padding) : (40, 64) -> (50, 64)
		# 6) Maxpooling : (50, 64) -> (50, 32)
		# 7) Conv 1d + ReLu (60 filtres de taille 3 avec padding) : (50, 32) -> (60, 32)
		# 8) Maxpooling : (60, 32) -> (60, 16)
		# 9) Flatten : (60, 16) -> (960)
		# 10) Linear + ReLu : (960) -> (64)
		# 11) Linear + sigmoid : (64) -> (1)
  
		self.nn = nn.Sequential(
			nn.Conv1d(70, 30, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(30, 40, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(40, 50, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(50, 60, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Flatten(),
			nn.Linear(60*16, 64),
			nn.ReLU(),
			nn.Linear(64, 1),
			nn.Sigmoid()
		)
  
	def forward(self, x):
		return self.nn(x)




# CNN Architecture in the paper (not used)
# Embedding : (max_seq_length=256, nb_char_dic=70) -> (128, 16)	???
# Conv + ReLu : (128, 16) -> (64, 1, 8)
# Maxpooling : (64, 1, 8) -> (32, 1, 8)
# Conv + ReLu : (32, 1, 8) -> (16, 1, 16)
# Maxpooling : (16, 1, 16) -> (8, 1, 16)
# Conv + ReLu : (8, 1, 16) -> (8, 1, 32)
# Maxpooling : (8, 1, 32) -> (1, 1, 32)
# Linear + ReLu : (1, 1, 32) -> (32)
# Linear + sigmoid : (32) -> (1)