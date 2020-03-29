import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from itertools import chain

class CancerDatasetWrapper:
	class __CancerDatasetWrapper:
		"""
		A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
		"""
		def __init__(self, cv_iters):
			"""
			create df for features and labels
			remove samples that are not shared between the two tables
			"""

			self.cv_iters = cv_iters
			self.labels = pd.read_csv("TCGA-BRCA.survival.tsv",delimiter='\t',encoding='utf-8') 
			#filter LumA donors
			donor = pd.read_csv("TCGA_PAM50.txt",delimiter='\t',encoding='utf-8') 
			donor = donor[donor['PAM50_genefu'] == 'LumA']
			donor = donor['submitted_donor_id']

			self.labels = self.labels[self.labels['_PATIENT'].isin(donor)]
			label_sample_list = list(self.labels['sample'])

			self.features = pd.read_csv("TCGA-BRCA.methylation450.tsv",delimiter='\t',encoding='utf-8') 
			self.features = self.features.dropna().reset_index(drop=True)
			self.features = self.features.drop([list(self.features.columns.values)[0]], axis=1)

			feature_sample_list = list(self.features.columns.values)
			samples = list(set(label_sample_list) & set(feature_sample_list))
			
			#only keep LumA samples to limit the memory usage
			self.features = self.features[samples]
			self.labels = self.labels[self.labels['sample'].isin(samples)]

			self.shuffle()

		def label(self, key):
			"""
			Args: 
				key:(string) the sample key	
			Returns:
				label to the life and death of patient
			"""
			return self.labels[self.labels['sample'] == key]['OS']

		def shuffle(self):
			"""
			categorize sample ID by label
			"""
			#keys to feature where label is 1
			self.ones = list(self.labels[self.labels['OS']==1]['sample'])
			index = np.arange(len(self.ones))
			np.random.shuffle(index)
			self.ones = [self.ones[index[i]] for i in range(index.shape[0])]
			self.ones = [self.ones[int(len(self.ones)/self.cv_iters)*i: int(len(self.ones)/self.cv_iters)*(i+1)] for i in range(self.cv_iters)]

			#keys to feature where label is 0
			self.zeros = list(self.labels[self.labels['OS']==0]['sample'])
			index = np.arange(len(self.zeros))
			np.random.shuffle(np.arange(len(self.zeros)))
			self.zeros = [self.zeros[index[i]] for i in range(index.shape[0])]
			self.zeros = [self.zeros[int(len(self.zeros)/self.cv_iters)*i: int(len(self.zeros)/self.cv_iters)*(i+1)] for i in range(self.cv_iters)]

			#index of valication set
			self.CVindex = 0

		def next(self):
			'''
			rotate to the next cross validation process
			'''
			if self.CVindex < self.cv_iters-1:
				self.CVindex += 1
			else:
				self.CVindex = 0


	instance = None
	def __init__(self, cv_iters, shuffle = 0):
		if not CancerDatasetWrapper.instance:
			CancerDatasetWrapper.instance = CancerDatasetWrapper.__CancerDatasetWrapper(cv_iters)
		elif shuffle:
			CancerDatasetWrapper.instance.shuffle()

	def __getattr__(self, name):
		return getattr(self.instance, name)

	def features(self, key):
		"""
		Args: 
			key:(string) value from dataset	
		Returns:
			features in list	
		"""
		return np.array(list(CancerDatasetWrapper.instance.features[key]))

	def label(self, key):
		"""
		Args: 
			key:(string) the sample key/id	
		Returns:
			label to the life and death of patient
		"""
		return np.array(list(CancerDatasetWrapper.instance.label(key)))

	def next(self):
		CancerDatasetWrapper.instance.next()

	def shuffle(self):
		CancerDatasetWrapper.instance.shuffle()

	def __trainSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of trainning set
		"""

		ind = list(range(CancerDatasetWrapper.instance.cv_iters))
		ind = np.delete(ind, CancerDatasetWrapper.instance.CVindex)

		trainSet = list(chain(*[CancerDatasetWrapper.instance.zeros[i] for i in ind]))+list(chain(*[CancerDatasetWrapper.instance.ones[i] for i in ind]))

		return trainSet
	
	def __valSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of validation set
		"""

		valSet = CancerDatasetWrapper.instance.zeros[CancerDatasetWrapper.instance.CVindex] + CancerDatasetWrapper.instance.ones[CancerDatasetWrapper.instance.CVindex]

		return valSet

	def __fullSet(self):
		"""
		Returns:
			dataset: (np.ndarray) array of key/id of full dataset
		"""

		fullset = list(chain(*CancerDatasetWrapper.instance.zeros))+list(chain(*CancerDatasetWrapper.instance.ones))

		return fullset

	def getDataSet(self, dataSetType = 'train'):
		"""
		Args: 
			dataSetType: (string) 'train' or 'val'	
		Returns:
			dataset: (np.ndarray) array of key/id of data set
		"""

		if dataSetType == 'train':
			return self.__trainSet()

		if dataSetType == 'val':
			return self.__valSet()

		return self.__fullSet()
		


class CancerDataset(Dataset):
	"""
	A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
	"""
	def __init__(self, dataSetType, CV_iters):
		"""
		initialize DatasetWrapper
		"""
		self.DatasetWrapper = CancerDatasetWrapper(CV_iters)

		self.samples = self.DatasetWrapper.getDataSet(dataSetType)

	def __len__(self):
		# return size of dataset
		return len(self.samples)


	def __getitem__(self, idx):
		"""
		Fetch feature and labels from dataset using index of the sample.

		Args:
		    idx: (int) index of the sample

		Returns:
		    feature: (Tensor) feature
		    label: (int) corresponding label of sample
		"""
		sample = self.samples[idx]
		return Tensor(self.DatasetWrapper.features(sample)), self.DatasetWrapper.label(sample)


def fetch_dataloader(types, params):
	"""
	Fetches the DataLoader object for each type in types.

	Args:
	types: (list) has one or more of 'train', 'val'depending on which data is required '' to get the full dataSet
	params: (Params) hyperparameters

	Returns:
	data: (dict) contains the DataLoader object for each type in types
	"""
	dataloaders = {}
	
	if len(types)>0:
		for split in types:
			if split in ['train', 'val']:
				dl = DataLoader(CancerDataset(split, params.CV_iters), batch_size=params.batch_size, shuffle=True,
					num_workers=params.num_workers,
					pin_memory=params.cuda)

				dataloaders[split] = dl
	else:
		dl = DataLoader(CancerDataset('',params.CV_iters), batch_size=params.batch_size, shuffle=True,
			num_workers=params.num_workers,
			pin_memory=params.cuda)

		return dl

	return dataloaders

def get_next_CV_set(CV_iters):
	DatasetWrapper = CancerDatasetWrapper(CV_iters)
	DatasetWrapper.next()
