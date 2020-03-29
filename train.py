import os
import argparse
from tqdm import tqdm
from numpy import isnan
import torch
import time
import logging
import gc

import torch.backends.cudnn as cudnn
from Evaluation_Matix import *
from utils import *
from data_loader import fetch_dataloader, get_next_CV_set

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(description='PyTorch Deep Neural Net Training')
parser.add_argument('--train', default = False, type=str2bool, 
			help="specify whether train the model or not (default: False)")
parser.add_argument('--model_dir', default='Model', 
			help="Directory containing params.json")
parser.add_argument('--resume', default = True, type=str2bool, 
			help='path to latest checkpoint (default: True)')
parser.add_argument('--network', type=str, default='basemodel', 
			help='select network to train on.')
parser.add_argument('--log', default='warning', type=str,
			help='set logging level')


def train_model(args, params, loss_fn, model, optimizer, CViter):
	start_epoch = 0
	best_AUC = 0

	if args.resume:
		logging.info('Resuming Check point: {}'.format(args.resume))
		start_epoch, best_AUC, model, optimizer = resume_checkpoint(args, model, optimizer, CViter)

	logging.info("fetch_dataloader")
	dataloaders = fetch_dataloader(['train', 'val'], params) 

	for epoch in range(start_epoch, start_epoch + params.epochs):
		logging.warning(' Training Epoch: [{0}]'.format(epoch))
		train(dataloaders['train'], model, loss_fn, optimizer, epoch)

		# evaluate on validation set
		val_loss, AUC = get_AUC(validate(dataloaders['val'], model, loss_fn))
		logging.warning('    Loss {loss:.4f};  AUC {AUC:.4f}\n'.format(loss=val_loss, AUC=AUC))

		# remember best loss and save checkpoint
		is_best = best_AUC < AUC
		best_AUC = max(best_AUC, AUC)
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_AUC': best_AUC,
			'optimizer' : optimizer.state_dict(),
			}, is_best, args, CViter)
	_, _, model, _ = resume_checkpoint(args, model, optimizer, CViter, '_model_best')
	save_ROC(args, CViter, validate(dataloaders['val'], model, loss_fn)[1])

	

def train(train_loader, model, loss_fn, optimizer, epoch):
	losses = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	
	with tqdm(total=len(train_loader)) as t:
		for i, (datas, label) in enumerate(train_loader):
			# measure data loading time
			logging.info("    Sample {}:".format(i))

			logging.info("        Loading Varable")
			input_var = torch.autograd.Variable(datas.cuda())
			label_var = torch.autograd.Variable(label.cuda()).double()

			# compute output
			logging.info("        Compute output")
			output = model(input_var).double()


			# measure record cost
			cost = loss_fn(output, label_var, (1, 1))
			assert not isnan(cost.cpu().data.numpy()),  "Gradient exploding, Loss = {}".format(cost.cpu().data.numpy())
			losses.update(cost.cpu().data.numpy(), len(datas))

			# compute gradient and do SGD step
			logging.info("        Compute gradient and do SGD step")
			optimizer.zero_grad()
			cost.backward()
			optimizer.step()

			gc.collect()
			t.set_postfix(loss='{:05.3f}'.format(losses()))
			t.update()

def validate(val_loader, model, loss_fn):
	logging.info("Validating")
	logging.info("Initializing measurement")
	losses = AverageMeter()
	outputs = [np.array([]), np.array([])]

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (datas, label) in enumerate(val_loader):
		logging.info("    Sample {}:".format(i))
		logging.info("        Loading Varable")
		input_var = torch.autograd.Variable(datas.cuda())
		label_var = torch.autograd.Variable(label.cuda()).double()

		# compute output
		logging.info("        Compute output")
		output = model(input_var).double()
		outputs[0] = np.concatenate((outputs[0], output.cpu().data.numpy().flatten()))
		outputs[1] = np.concatenate((outputs[1], label_var.cpu().data.numpy().flatten()))
		loss = loss_fn(output, label_var, (1, 1))
		assert not isnan(loss.cpu().data.numpy()),  "Overshot loss, Loss = {}".format(loss.cpu().data.numpy())
		# measure record cost
		losses.update(loss.cpu().data.numpy(), len(datas))
	
	return losses.sum, outputs


def main():
	args = parser.parse_args()
	params = set_params(args.model_dir, args.network)

	# Set the logger
	set_logger(args.model_dir, args.network, args.log)
	
	# create model
	import model_loader
	model = model_loader.loadModel(args.network, params.dropout_rate)
	model.cuda()

	# define loss function and optimizer
	loss_fn = UnevenWeightBCE_loss
	optimizer = torch.optim.Adam(model.parameters(), params.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=params.weight_decay, amsgrad=False)
	cudnn.benchmark = True

	if args.train:
		for CViter in range(params.CV_iters):
			logging.warning('Cross Validation on iteration {}'.format(CViter+1))			
			train_model(args, params, loss_fn, model, optimizer, CViter)
			get_next_CV_set(params.CV_iters)
	else: 
		best_AUC = 0
		logging.info('ploting ROC on full dataset for {}'.format(args.network))
		for CViter in range(params.CV_iters):
			params.start_epoch, AUC, model, optimizer = resume_checkpoint(args, model, optimizer, CViter, '_model_best')
			if best_AUC < AUC:
				best_AUC = AUC
				best_model = model

		save_ROC(args, '_Full_dataset_', outputs= validate(fetch_dataloader([], params), best_model, loss_fn)[1], display = True) #validate model on the full dataset, display ROC curve

		

if __name__ == '__main__':
	main()
