import torch.sigmoid as sigmoid
import torch.nn as nn

import model.functions

class Base_Model(nn.Module):
	def __init__(self, channels):
		"""
		Args:
		    params: (Params) contains num_channels
		"""
		super(Base_Model, self).__init__()
		self.fc = nn.Linear(channels, channels) 

	def forward(self, x):
		"""
		This function defines how we use the components of our network to operate on an input batch.

		Args:
		    X: (Variable) features.

		Returns:
		    out: (Variable) dimension batch_size x 1 with the log probabilities for the prediction.

		Note: the dimensions after each step are provided
		"""
		x = self.fc(x)
		x = sigmoid(x)

		return x

