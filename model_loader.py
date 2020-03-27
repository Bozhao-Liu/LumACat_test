import os
import sys
import torch
import logging

def loadModel(netname = 'basemodel', channels = 1):
    Netpath = 'Model'
    Netfile = os.path.join(Netpath, netname)
    Netfile = os.path.join(Netfile, netname + '.py')
    assert os.path.isfile(Netfile), "No python file found for {}, (file name is case sensitive)".format(Netfile)
    netname = netname.lower()
    if netname == 'basemodel': 
        model = loadBaseModel(channels)
    else:
        logging.warning("No model with the name {} found, please check your spelling.".format(netname))
        logging.warning("Net List:")
        logging.warning("    basemodel")
        sys.exit()
    return model
    
def loadBaseModel(channels):
    from Model.basemodel.basemodel import Base_Model
    print("Loading Base Model")
    return Base_Model(channels)
    


