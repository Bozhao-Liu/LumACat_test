import pandas as pd
import numpy as np
from itertools import chain
from random import shuffle
import torch

'''
labels = pd.read_csv("TCGA-BRCA.survival.tsv",delimiter='\t',encoding='utf-8') 
#filter LumA donors
donor = pd.read_csv("TCGA_PAM50.txt",delimiter='\t',encoding='utf-8') 
donor = donor[donor['PAM50_genefu'] == 'LumA']
donor = donor['submitted_donor_id']

labels = labels[labels['_PATIENT'].isin(donor)]

features = pd.read_csv("TCGA-BRCA.methylation450.tsv",delimiter='\t',encoding='utf-8') 
feautres = features.dropna().reset_index(drop=True)
feautres = feautres.drop([list(feautres.columns.values)[0]], axis=1)
print(len(feautres[list(feautres.columns.values)[0]]))

feature_sample_list = list(feautres.columns.values)

ones = list(labels[labels['OS']==1]['sample'])
zeros = list(labels[labels['OS']==0]['sample'])
print(len(ones),len(zeros))'''
pt_ex_int_tensor = (torch.rand(2, 3, 4) * 100)
print(pt_ex_int_tensor)
np_ex_int_mda = pt_ex_int_tensor.numpy()
np_ex_int_mda.shape
print(np_ex_int_mda)
