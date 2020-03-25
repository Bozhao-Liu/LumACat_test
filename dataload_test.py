import pandas as pd
import numpy as np

features = pd.read_csv("TCGA-BRCA.methylation450.tsv",delimiter='\t',encoding='utf-8') 
feautres = features.dropna().reset_index(drop=True)
feautres = feautres.drop([list(feautres.columns.values)[0]], axis=1)

labels = pd.read_csv("TCGA-BRCA.survival.tsv",delimiter='\t',encoding='utf-8') 
#filter LumA donors
donor = pd.read_csv('TCGA_PAM50.txt', delimiter='\t',encoding='utf-8') 
donor = donor[donor['PAM50_genefu'] == 'LumA']
donor = donor['submitted_donor_id']
labels = labels[labels['_PATIENT'].isin(donor)] 
label_sample_list = list(labels['sample'])
feature_sample_list = list(feautres.columns.values)[1:]
samples = list(set(label_sample_list) & set(feature_sample_list))

feautres = feautres[samples]
labels = labels[labels['sample'].isin(samples)]
print(list(feautres.columns.values))
