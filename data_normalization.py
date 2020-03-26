import pandas as pd
import numpy as np

print('Reading original tsv')
data = pd.read_csv("TCGA-BRCA.methylation450.tsv",delimiter='\t',encoding='utf-8') 
data = data.dropna().reset_index(drop=True)
data = data.drop([list(data.columns.values)[0]], axis=1)

print('Geting normalized data')
maxvalue = np.max(data.values, axis = 1)
maxvalue = np.reshape(maxvalue, (-1, 1))
values = data.values/maxvalue
values = np.transpose(values)

print('Composing new data to dictionary')
newdata = {}
for i in range(len(list(data.columns.values))):
	newdata[data.columns.values[i]]= values[i]

print('Constructing and save normalized table file')
df = pd.DataFrame(newdata)
df.to_csv ('filterd_methylation450.csv', index = False, header=True)

print('Naormalization finished')


