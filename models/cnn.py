import torch.nn as nn


class CNNUrl(nn.Module):

	def __init__(self) -> None:
		super().__init__()

		################## ARCHITECTURE 1 ##################
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
  
		self.nn1 = nn.Sequential(
			nn.Conv1d(70, 128, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(128, 256, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(256, 512, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Conv1d(512, 1024, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=2),
			nn.Flatten(),
			nn.Linear(1024*16, 128),
			nn.ReLU(),
			nn.Linear(128, 1),
			nn.Sigmoid()
		)
  
		################## ARCHITECTURE 2 ##################
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