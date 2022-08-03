import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
from transformers import AutoModelForSequenceClassification, \
    AutoTokenizer

""" This file imports the ClimateFinanceBERT classifiers to predict climate relevance and categories for the crs dataset"""

pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 10)
# specify GPU
device = torch.device("cuda")

#import data
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
print(dir_path)


# Import the labeling dictionary
import json
###FOr importing the dictionaries
def load_dict(filename):
    with open(filename) as f:
        file = json.loads(f.read())
    return file
label_dict = load_dict(dir_path+'/Code/saved_models/reverse_dictionary_classes')
print(label_dict)


"""MODEL """
base_model = 'climatebert/distilroberta-base-climate-f'
# import BERT-base pretrained model
relevance_classifier = AutoModelForSequenceClassification.from_pretrained(base_model,
                                                              num_labels=2,
                                                                )

multiclass = AutoModelForSequenceClassification.from_pretrained(base_model,
                                                              num_labels=len(label_dict),
                                                                )

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)


class BERT_Arch(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.bert = model

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model
      output = self.bert(sent_id, attention_mask=mask, return_dict=False)

      # apply softmax activation
      x = self.softmax(output[0])
      return x


#initialize relevance_classifier
relevance_classifier = BERT_Arch(relevance_classifier)
# push the model to GPU
relevance_classifier = relevance_classifier.to(device)
#Load trained weights
relevance_classifier.load_state_dict(torch.load(
    dir_path+'/Code/saved_models/saved_weights_relevance.pt'))

#initialize multiclass_model
multiclass = BERT_Arch(multiclass)
# push the model to GPU
multiclass = multiclass.to(device)
#Load trained weights
multiclass.load_state_dict(torch.load(
    dir_path+'/Code/saved_models/saved_weights_multiclass.pt'))



def tokenize(sentence):
    """Tokenization"""
    token = tokenizer.batch_encode_plus(
        sentence,
        max_length = 150,
        pad_to_max_length=True,
        truncation=True
    )

    test_seq = torch.tensor(token['input_ids'])
    test_mask = torch.tensor(token['attention_mask'])
    return test_seq, test_mask




"""Import the data in chunks"""

def csv_import(name, delimiter="|", chunksize=10 ** 3):
    x = pd.read_csv(name, encoding='utf8', low_memory=False, delimiter=delimiter, chunksize=chunksize,
                    dtype={'text': str,
                           "USD_Disbursement": float
                           }
                    )
    return x

#List to append chunks
chunk_list = []

i=0
# import aid data set
for chunk in csv_import(dir_path+'/Data/df_preprocessed.csv'):
    print(chunk.shape)
    chunk['DonorType']= 'Multilateral Donor Organization'
    chunk['DonorType'][chunk.DonorCode < 807]= 'Donor Country'
    chunk['DonorType'][chunk.DonorCode > 1600]= 'Private Donor'
    chunk['DonorType'][chunk.DonorCode == 104]= 'Multilateral Donor Organization'
    chunk['DonorType'][chunk.DonorCode.isin([820])]= 'Donor Country'

    chunk = chunk[chunk.DonorType == 'Donor Country']
    print('binary scope:', chunk.shape)
    chunk['prediction_criterium']=0
    chunk['prediction_criterium'][chunk.text.str.split().str.len() > 3]=1

    #define default of climate_relevance -->0
    chunk['climate_relevance'] = 0
    if chunk[chunk['prediction_criterium']==1].shape[0]>0:
        print('predict relevance')
        text_seq, text_mask = tokenize(chunk.text[chunk.prediction_criterium == 1].to_list())

        # get predictions for new data
        with torch.no_grad():
          preds = relevance_classifier(text_seq.to(device), text_mask.to(device))
          preds = preds.detach().cpu().numpy()

        pred_relevance = np.argmax(preds, axis = 1)

        chunk['climate_relevance'][chunk.prediction_criterium == 1]=pred_relevance


    """MULTICLASS"""
    print('number of documents ', chunk.text.shape[0])
    print('number of relevant documents ', chunk.text[chunk.climate_relevance == 1].shape[0])
    # mask all texts that are relevant
    chunk['climate_class_number'] = 500

    if chunk.text[chunk.climate_relevance == 1].shape[0]>0:
        text_seq, text_mask = tokenize(chunk.text[chunk.climate_relevance == 1].to_list())

        # get predictions for test data
        with torch.no_grad():
          preds = multiclass(text_seq.to(device), text_mask.to(device))
          preds = preds.detach().cpu().numpy()

        pred_class = np.argmax(preds, axis = 1)

        chunk['climate_class_number'][chunk.climate_relevance == 1] = pred_class
        chunk['climate_class'] = chunk['climate_class_number'].astype(str).replace(label_dict)

    else:
        chunk['climate_class'] = '500'


    #append chunk to list
    chunk_list.append(chunk)

    i = i+1
    print(i)

    del chunk

df = pd.concat(chunk_list)

print(df.shape)
print(df[df.climate_class == 'Adaptation'].shape[0])
adaptation = df[df.climate_class == 'Adaptation']
adaptation = adaptation[['USD_Disbursement','Year']].groupby('Year').sum().reset_index()
adaptation.to_csv(dir_path+'/Data/adaptation.csv', encoding='utf8', index=False, header=True, sep=',')

list_mitigation = ["Solar-energy","Biofuel-energy", "Other-mitigation-projects","Geothermal-energy","Marine-energy",
                    "Renewables-multiple", "Hydro-energy", "Energy-efficiency", "Wind-energy"]
mitigation = df[df.climate_class.isin(list_mitigation)]
mitigation = mitigation[['USD_Disbursement','Year']].groupby('Year').sum().reset_index()
mitigation.to_csv(dir_path+'/Data/mitigation.csv', encoding='utf8', index=False, header=True, sep=',')

list_nature = ["Nature_conservation","Sustainable-land-use","Other-environment-protection-projects","Biodiversity"]
nature = df[df.climate_class.isin(list_nature)]
nature = nature[['USD_Disbursement','Year']].groupby('Year').sum().reset_index()
nature.to_csv(dir_path+'/Data/nature.csv', encoding='utf8', index=False, header=True, sep=',')

df.to_csv(dir_path+'/Data/climate_finance_total.csv', encoding='utf8', index=False, header=True, sep='|')
