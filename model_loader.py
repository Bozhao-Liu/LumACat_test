import os
import sys
import torch
import logging

def loadModel(netname = 'basemodel',dropout_rate = 0.5, channels = 1):
    Netpath = 'Model'
    Netfile = os.path.join(Netpath, netname)
    Netfile = os.path.join(Netfile, netname + '.py')
    assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
    netname = netname.lower()
    if netname == 'basemodel': 
        model = loadBaseModel(channels, dropout_rate)
    elif netname == 'fc_1h':
        model = loadFC_1HModel(channels, dropout_rate)
    elif netname == 'fc_2h':
        model = loadFC_2HModel(channels, dropout_rate)
    else:
        logging.warning("No model with the name {} found, please check your spelling.".format(netname))
        logging.warning("Net List:")
        logging.warning("    basemodel")
        logging.warning("    FC_1H")
        sys.exit()
    return model
    
def loadBaseModel(channels, dropout_rate):
    from Model.basemodel.basemodel import Base_Model
    logging.warning("Loading Base Model")
    return Base_Model(channels, dropout_rate)

def loadFC_1HModel(channels, dropout_rate):
    from Model.FC_1H.FC_1H import FC_1H
    logging.warning("Loading 1 hidden layer model")
    return FC_1H(channels, dropout_rate)
    
def loadFC_2HModel(channels, dropout_rate):
    from Model.FC_2H.FC_2H import FC_2H
    logging.warning("Loading 2 hidden layer model")
    return FC_2H(channels, dropout_rate)
    

