import os
import sys
import torch
import logging

def loadModel(channels, netname = 'basemodel'):
    Netpath = 'Model'
    Netfile = os.path.join(Netpath, netname + '.py')
    assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
    netname = netname.lower()
    if netname == 'basemodel': 
        model, version = loadBaseModel(channels)
    else:
        logging.warning("No model with the name {} found, please check your spelling.".format(netname))
        logging.warning("Net List:")
        logging.warning("    basemodel")
        sys.exit()
    return model, version
    
def loadBaseModel(channels):
    from Model.base_model import Base_model
    print("Loading Base Model")
    return Base_Model(channels), ''
    

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
