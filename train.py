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
parser.add_argument('--resume', default=False, type=str, metavar='PATH', 
			help='path to latest checkpoint (default: none)')
parser.add_argument('--network', type=str, default='base_model', 
			help='select network to train on. (no default, must be specified)')
parser.add_argument('--print-freq', '-p', default=50, type=int, metavar='N', 
			help='print frequency (default: 10)')
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

def main():
	args = parser.parse_args()
	params = set_params(args.model_dir, args.network)
		
	dataloaders = fetch_dataloader(['train', 'val'], params) 

	# Set the logger
	utils.set_logger(os.path.join(json_path, 'train.log'), args.log)
	

if __name__ == '__main__':
	main()
