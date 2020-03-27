import os
import argparse
from tqdm import tqdm
import torch
import gc

from Evaluation_Matix import *
from utils import *
from data_loader import fetch_dataloader, get_next_CV_set

parser = argparse.ArgumentParser(description='PyTorch Deep Neural Net Training')
parser.add_argument('--train', default=False,  type=bool, 
			help="specify whether train the model or not")
parser.add_argument('--model_dir', default='Model', 
			help="Directory containing params.json")
parser.add_argument('--resume', default=True, type=bool, 
			help='path to latest checkpoint (default: True)')
parser.add_argument('--network', type=str, default='basemodel', 
			help='select network to train on.')
parser.add_argument('--log', default='warning', type=str,
			help='set logging level')


def train_model(args, params, loss_fn, model, optimizer, CViter):
	if args.resume:
		start_epoch, best_loss, model, optimizer = resume_checkpoint(args.model_dir, args.network, model, optimizer, CViter)

	dataloaders = fetch_dataloader(['train', 'val'], params) 

	for epoch in range(params.start_epoch, params.start_epoch + params.epochs):
		logging.info("Epoch {}:".format(epoch))
		train(dataloaders['train'], model, loss_fn, optimizer)

		# evaluate on validation set
		val_loss = validate(dataloaders['val'], model, loss_fn)

		# remember best loss and save checkpoint
		is_best = val_loss > best_loss
		best_loss = max(val_loss, best_loss)
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_loss': best_loss,
			'optimizer' : optimizer.state_dict(),
			}, is_best, args.model_dir, args.network, CViter)
		if is_best:
			save_AUC(params, args.model_dir, args.network, CViter, output.data, label)

	get_next_CV_set()

def train(train_loader, model, loss, optimizer, epoch, threshold = 0.5):
	logging.info("Epoch {}:".format(epoch))
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	with tqdm(total=len(train_loader)) as t:
		for i, (datas, label, _) in enumerate(train_loader):
			# measure data loading time
			logging.info("    Sample {}:".format(i))
			data_time.update(time.time() - end)

			logging.info("        Loading Varable")
			input_var = torch.autograd.Variable(datas.cuda())
			label_var = torch.autograd.Variable(label.cuda()).double()

			# compute output
			logging.info("        Compute output")
			output = model(input_var).double()


			# measure accuracy and record cost
			cost = loss(output, label_var, (1, 1))
			losses.update(cost.data, len(datas))

			# compute gradient and do SGD step
			logging.info("        Compute gradient and do SGD step")
			optimizer.zero_grad()
			cost.backward()
			optimizer.step()

			# measure elapsed time
			logging.info("        Measure elapsed time")
			batch_time.update(time.time() - end)
			end = time.time()


			gc.collect()
			t.set_postfix(loss='{:05.3f}'.format(losses()))
			t.update()

		logging.warning('Epoch: [{0}][{1}/{2}]\n'
				'    Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
				'    Data {data_time.val:.3f} ({data_time.avg:.3f})\n'
				'    Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(
				epoch, i, len(train_loader), batch_time=batch_time,
				data_time=data_time, loss=losses))

def validate(val_loader, model, loss, threshold = 0.5):
	logging.info("Validating")
	logging.info("Initializing measurement")
	batch_time = AverageMeter()
	losses = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (datas, label, _) in enumerate(val_loader):
		logging.info("    Sample {}:".format(i))
		logging.info("        Loading Varable")
		input_var = torch.autograd.Variable(datas.cuda())
		label_var = torch.autograd.Variable(label.cuda()).double()

		# compute output
		logging.info("        Compute output")
		output = model(input_var).double()
		cost = loss(output, label_var, (1, 1))

		# measure record cost
		losses.update(cost.data, len(datas))

		# measure elapsed time
		logging.info("        Measure elapsed time")
		batch_time.update(time.time() - end)
		end = time.time()

	logging.warning('Test: [{0}/{1}]\n'
			'    Time {batch_time.val:.3f} ({batch_time.avg:.3f})\n'
			'    Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(
			i, len(val_loader), batch_time=batch_time, loss=losses))


	return losses.sum


def main():
	args = parser.parse_args()

	params = set_params(args.model_dir, args.network)

	# Set the logger
	set_logger(args.model_dir, args.network, args.log)
	
	# create model
	logging.warning("Loading Model")
	import model_loader
	model = model_loader.loadModel(args.network)
	model.cuda()

	# define loss function and optimizer
	loss_fn = UnevenWeightBCE_loss
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

		

if __name__ == '__main__':
	main()
