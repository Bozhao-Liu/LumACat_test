import pandas as pd
import numpy as np
from itertools import chain
from random import shuffle


labels = pd.read_csv("TCGA-BRCA.survival.tsv",delimiter='\t',encoding='utf-8') 
#filter LumA donors
donor = pd.read_csv("TCGA_PAM50.txt",delimiter='\t',encoding='utf-8') 
donor = donor[donor['PAM50_genefu'] == 'LumA']
donor = donor['submitted_donor_id']
print(len(donor))

labels = labels[labels['_PATIENT'].isin(donor)]

features = pd.read_csv("TCGA-BRCA.methylation450.tsv",delimiter='\t',encoding='utf-8') 
feautres = features.dropna().reset_index(drop=True)
print(list(feautres.columns.values))
feautres = feautres.drop([list(feautres.columns.values)[0]], axis=1)
print(list(feautres.columns.values))

feature_sample_list = list(feautres.columns.values)

ones = list(labels[labels['OS']==1]['sample'])
'''index = np.arange(len(ones))
np.random.shuffle(index)
ones = [ones[index[i]] for i in range(index.shape[0])]
ones = [ones[int(len(ones)/10)*i: int(len(ones)/10)*(i+1)] for i in range(10)]'''

zeros = list(labels[labels['OS']==0]['sample'])
'''
index = np.arange(len(zeros))
np.random.shuffle(np.arange(len(zeros)))
zeros = [zeros[index[i]] for i in range(index.shape[0])]
zeros = [zeros[int(len(zeros)/10)*i: int(len(zeros)/10)*(i+1)] for i in range(10)]'''
print(len(ones),len(zeros))
'''
ind = np.arange(10)
ind = np.delete(ind, 3)
print(ind)
print([zeros[i] for i in ind])

trainSet = list(chain(*[zeros[i] for i in ind]))+list(chain(*[ones[i] for i in ind]))
index = np.arange(len(trainSet))
np.random.shuffle(index)
trainSet = [trainSet[index[i]] for i in range(len(index))]
print(trainSet)

valSet = zeros[3] + ones[3]
index = np.arange(len(valSet))
np.random.shuffle(index)
valSet = [valSet[index[i]] for i in range(len(index))]
print(valSet)'''
