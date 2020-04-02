import torch.nn as nn


class FC_3H(nn.Module):
	def __init__(self, channels, dropout_rate):
		"""
		Args:
		    params: (Params) contains num_channels
		"""
		super(FC_3H, self).__init__()
		self.fc = nn.Sequential(
			nn.Dropout(dropout_rate), 
			nn.Linear(363791, 1024),
			nn.Dropout(dropout_rate),
			nn.Linear(1024, 1024),
			nn.Dropout(dropout_rate),
			nn.Linear(1024, 1024),
			nn.Dropout(dropout_rate),
			nn.Linear(1024, channels),
			nn.Sigmoid())

	def forward(self, x):
		"""
		This function defines how we use the components of our network to operate on an input batch.

		Args:
		    X: (Variable) features.

		Returns:
		    out: (Variable) dimension batch_size x 1 with the log probabilities for the prediction.

		Note: the dimensions after each step are provided
		"""
		return self.fc(x)

