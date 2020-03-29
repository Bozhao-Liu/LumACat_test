import os
import torch
import logging
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __call__(self):
        return self.avg

def UnevenWeightBCE_loss(outputs, labels, weights = (1, 1)):
	'''
	Cross entropy loss with uneven weigth between positive and negative result to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(weights[0]*torch.mul(labels[:, i],torch.log(outputs[:, i])), weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i])))) for i in range(outputs.shape[1])]
	return -torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def Exp_UEW_BCE_loss(outputs, labels, weights = (1, 1)):
	'''
	Cross entropy loss with uneven weigth between positive and negative result, add exponential function to positive to manually adjust precision and recall
	'''
	loss = [torch.sum(torch.add(weights[0]*torch.exp(-torch.mul(labels[:, i],torch.log(outputs[:, i])))-1, -weights[1]*torch.mul(1 - labels[:, i],torch.log(1 - outputs[:, i])))) for i in range(outputs.shape[1])]
	return torch.stack(loss, dim=0).sum(dim=0).sum(dim=0)

def save_ROC(args, CViter, outputs, display = False):
	AUC_png_file = os.path.join(args.model_dir, args.network)
	AUC_png_file = os.path.join(AUC_png_file, 'AUC')
	if not os.path.isdir(AUC_png_file):
		os.mkdir(AUC_png_file)
	AUC_png_file = os.path.join(AUC_png_file, args.network+ str(CViter) + '.PNG')
	AUC = 0
	TP_rates = []
	FP_rates = []
	FP_rate_pre = 1
	TP_rate_pre = 1
	logging.warning('Creating ROC image for {} \n'.format(args.network))
	for i in tqdm(np.linspace(0, 1, 51)):
		results = outputs[0]>i
		TP = np.sum((results+outputs[1])==2, dtype = float)
		FN = np.sum(results<outputs[1], dtype = float)
		TP_rate = TP/(TP + FN + 1e-8)
		TP_rates.append(TP_rate)
		FP = np.sum(results>outputs[1], dtype = float)
		TN = np.sum((results+outputs[1])==0, dtype = float)
		FP_rate = FP/(FP + TN + 1e-8)
		FP_rates.append(FP_rate)
		AUC += (TP_rate_pre+TP_rate)*(FP_rate_pre-FP_rate)/2.0
		FP_rate_pre = FP_rate
		TP_rate_pre = TP_rate
		
	plt.clf()
	plt.plot(FP_rates,TP_rates)
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.title('{} ROC on validation set NO.{}, AUC: {}'.format(args.network, CViter, AUC))
	if display:
		logging.warning('    Displaying ROC \n')
		plt.show()
	else:
		logging.warning('    Saving ROC plot to {}\n'.format(AUC_png_file))
		plt.savefig(AUC_png_file)

def get_AUC(output):
	outputs = output[1] #outputs[0] as predicted probs, outputs[1] as labels
	AUC = 0
	FP_rate_pre = 1
	TP_rate_pre = 1
	for i in np.linspace(0,1):
		results = outputs[0]>i
		TP = np.sum((results+outputs[1])==2, dtype = float)
		FN = np.sum(results<outputs[1], dtype = float)
		TP_rate = TP/(TP + FN)
		FP = np.sum(results>outputs[1], dtype = float)
		TN = np.sum((results+outputs[1])==0, dtype = float)
		FP_rate = FP/(FP + TN)
		AUC += (TP_rate_pre+TP_rate)*(FP_rate_pre-FP_rate)/2.0
		FP_rate_pre = FP_rate
		TP_rate_pre = TP_rate
	return output[0], AUC


