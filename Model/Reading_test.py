import csv
import numpy as np

import pandas as pd

#UserInfo.tsv
user_info=pd.read_csv('TCGA-BRCA.survival.tsv',delimiter='\t',encoding='utf-8')
print(user_info.columns.values) #file header
title = np.array(user_info['sample'])
print(title.shape)
dataset=pd.read_csv('TCGA-BRCA.methylation450.tsv',delimiter='\t',encoding='utf-8')
print('reading sample')
sample = np.array(dataset[title[1]])
print(sample.shape)
'''
sample = []
with open('TCGA-BRCA.survival.tsv') as tsvfile:
	sample = tsvfile[sample]
print(sample)

with open('TCGA-BRCA.methylation450.tsv') as tsvfile:
	reader = csv.DictReader(tsvfile, dialect='excel-tab')
	r = 0
	columns = {}
	for row in reader:
		for fieldname in reader.fieldnames:
			columns.setdefault(fieldname, []).append(row.get(fieldname))'''
