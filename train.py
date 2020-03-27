import os
import utils
import argparse
from tqdm import tqdm

import torch

from data_loader import fetch_dataloader

parser = argparse.ArgumentParser(description='PyTorch Deep Neural Net Training')
parser.add_argument('--train', default=False,  type=bool, 
			help="specify whether train the model or not")
parser.add_argument('--model_dir', default='Model', 
			help="Directory containing params.json")
parser.add_argument('--resume', default=True, type=bool, 
			help='path to latest checkpoint (default: True)')
parser.add_argument('--network', type=str, default='basemodel', 
			help='select network to train on.')
parser.add_argument('--print-freq', '-p', default=50, type=int, metavar='N', 
			help='print frequency (default: 50)')
parser.add_argument('--log', default='warning', type=str,
			help='set logging level')

def set_params(model_dir, network):
	params = utils.Params(model_dir, network)

	# use GPU if available
	params.cuda = torch.cuda.is_available()

	# Set the random seed for reproducible experiments
	torch.manual_seed(230)
	if params.cuda: 
		torch.cuda.manual_seed(230)

	return params

def resume_checkpoint(model_dir, network, start_epoch, best_loss, model, optimizer, CViter):
	checkpointfile = os.path.join(model_dir, network+ str(CViter) + '.pth.tar')
	if os.path.isfile(checkpointfile):
		#logging.warning("=> loaded checkpoint '{}' (epoch {})".format(checkpointfile, checkpoint['epoch']))
		logging.info("Loading checkpoint {}".format(checkpointfile))
		checkpoint = torch.load(checkpointfile)
		start_epoch = checkpoint['epoch']
		best_loss = checkpoint['best_loss']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		return start_epoch, best_loss, model, optimizer
	else:
		logging.warning("=> no checkpoint found at '{}'".format(checkpointfile))
		return 0, +inf, model, optimizer

def train_model(args, params, loss_fn, model, optimizer, CViter):
	if args.resume:
		start_epoch, best_loss, model, optimizer = resume_checkpoint(args.model_dir, args.network, model, optimizer, CViter)
	dataloaders = fetch_dataloader(['train', 'val'], params) 

	train_loader = dataloaders['train']
	val_loader = dataloaders['val']
	for epoch in range(params.start_epoch, params.start_epoch + params.epochs):
		logging.info("Epoch {}:".format(epoch))
		train(train_loader, model, loss_fn, optimizer)

		# evaluate on validation set
		val_loss = validate(val_loader, model, loss_fn)

		# remember best F1 and save checkpoint
		is_best = val_loss > (2*(best_F1[0]*best_F1[1])/(best_F1[0]+best_F1[1]))
		best_loss = max(val_loss, best_loss)
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_loss': best_loss,
			'optimizer' : optimizer.state_dict(),
			}, is_best, args.model_dir, args.network, CViter)
		if is_best:
			save_to_ini(params, args.model_dir, args.network, version, val_result, threshold = args.threshold)
    	validate(val_loader, model, loss, threshold = args.threshold)
	get_next_CV_set()

def get_next_CV_set():
	DatasetWrapper = CancerDatasetWrapper()
	DatasetWrapper.next()

def main():
	args = parser.parse_args()

	params = set_params(args.model_dir, args.network)

	# Set the logger
	utils.set_logger(os.path.join(json_path, 'train.log'), args.log)
	
	# create model
	logging.warning("Loading Model")
	model = model_loader.loadModel(args.network)
	model.cuda()

	# define loss function and optimizer
	loss_fn = model_loader.UnevenWeightBCE_loss
	optimizer = torch.optim.Adam(model.parameters(), params.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=params.weight_decay, amsgrad=False)
	cudnn.benchmark = True

	if args.train:
		for CViter in range(10):
			logging.warning('Cross Validation on iteration {}'.format(CViter+1))			
			train_model(args, params, loss_fn, model, optimizer, CViter)
			
	else: 
		best_Loss = +inf
		for CViter in range(10):
			params.start_epoch, loss, model, optimizer = resume_checkpoint(args.model_dir, args.network, model, optimizer, CViter)
			if loss < best_loss:
				best_model = model

		validate(fetch_dataloader([], params) , best_model, loss_fn) #validate model on the full dataset

def save_checkpoint(state, is_best, model_dir, network, CViter):
	checkpointfile = os.path.join(model_dir, network+ str(CViter) + '.pth.tar')
	torch.save(state, filename)
	if is_best:
		checkpointfile = os.path.join(model_dir, network+ str(CViter) + '_model_best.pth.tar')	
		torch.save(state, filename)
		

if __name__ == '__main__':
	main()
