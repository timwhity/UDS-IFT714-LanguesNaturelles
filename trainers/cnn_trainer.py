from .base_trainer import BaseTrainer
from tqdm import tqdm
import numpy as np
from typing import List
import torch

class CNNTrainer(BaseTrainer):
	def __init__(self, experiment, model, tokenizer, loss_fn, optimizer, scheduler, trainloader, validloader, testloader, classes, device, limit=None, max_seq_length=256) -> None:
		super().__init__(experiment, "cnn", model, tokenizer, loss_fn, optimizer, scheduler, trainloader, validloader, testloader, classes, device, limit)
		self.max_seq_length = max_seq_length
		self.char_dic = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'/\\|_@#$%ˆ&*˜‘+-=<>()[]{}\n"
		self.nb_char_dic = 70
		assert len(self.char_dic) == self.nb_char_dic

	
	def tokenizer(self, url: str) -> np.array:
		"""
		Input : string of the url
		Output : numpy array (nb_char_dic, max_seq_length)
		"""
		sparse_vector = np.zeros(( self.nb_char_dic, self.max_seq_length))
		nb_accepted_characters = 0
		for i, c in enumerate(url):
			if c in self.char_dic:
				sparse_vector[self.char_dic.index(c), nb_accepted_characters] = 1
				nb_accepted_characters += 1
			elif c.lower() in self.char_dic:
				sparse_vector[self.char_dic.index(c.lower()), nb_accepted_characters] = 1
				nb_accepted_characters += 1
			if nb_accepted_characters == self.max_seq_length:
				break
		return sparse_vector
	
	def tokenizer_batch(self, urls: List[str]) -> np.array:
		"""
		Input : list of urls
		Output : numpy array (len(urls), nb_char_dic, max_seq_length)
		"""
		return np.array([self.tokenizer(url) for url in urls])
	
	def predict(self, texts: List[str]):
		return self.model(self.tokenizer_batch(texts), probs=True)
	
	def train(self, eval_each: int = 0, epoch_title: str = "Epoch"):
		self.model.train()

		total_loss = 0.0
		total_correct = 0
		count_examples = 0
		count_batches = 0

		for batch_index, (inputs, targets) in enumerate(tqdm(self.trainloader, desc=epoch_title)):
			inputs = self.tokenizer_batch(inputs)
			inputs = inputs.to(self.device)
			targets = targets.to(self.device)

			self.optimizer.zero_grad()

			outputs = self.model(inputs)
			preds = outputs.argmax(dim=1)
			loss = self.loss_fn(outputs, targets)

			loss.backward()
			self.optimizer.step()
			self.scheduler.step()

			total_loss += loss.item()
			total_correct += (preds == targets).sum().item()
			count_examples += len(targets)
			count_batches += 1

			avg_loss = total_loss / count_batches
			accuracy = total_correct / count_examples
			self.metrics.update_train_metrics(avg_loss, accuracy, 0.0)

			if (eval_each > 0) and (batch_index % eval_each == 0):
				self.validate()
			if self.limit and (batch_index >= self.limit):
				break

		return self.metrics.get_metrics("train")
	
	def validate(self, test: bool = False):
		desc = "valid"
		dataloader = self.validloader
		if test:
			desc = "test"
			dataloader = self.testloader
			TP = 0
			TN = 0
			FP = 0
			FN = 0

		total_loss = 0.0
		total_correct = 0
		count_examples = 0
		count_batches = 0
		
		self.model.eval()

		with torch.no_grad():
			for batch_index, (inputs, targets) in enumerate(tqdm(dataloader, desc=desc)):
				inputs = self.tokenizer_batch(inputs)
				inputs = inputs.to(self.device)
				targets = targets.to(self.device)

				outputs = self.model(inputs)
				preds = outputs.argmax(dim=1)
				loss = self.loss_fn(outputs, targets)

				total_loss += loss.item()
				total_correct += (preds == targets).sum().item()
				count_examples += len(targets)
				count_batches += 1

				if test:
					TP += ((preds == 1) & (targets == 1)).sum().item()
					TN += ((preds == 0) & (targets == 0)).sum().item()
					FP += ((preds == 1) & (targets == 0)).sum().item()
					FN += ((preds == 0) & (targets == 1)).sum().item()

		avg_loss = total_loss / count_batches
		accuracy = total_correct / count_examples

		if test:
			self.metrics.update_test_metrics(avg_loss, accuracy, TP, TN, FP, FN)
		else:
			self.metrics.update_valid_metrics(avg_loss, accuracy)


	def test(self):
		return self.validate(test=True)




		
