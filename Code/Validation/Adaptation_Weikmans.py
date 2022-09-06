import pandas as pd
from helpers import Preprocess_text
import os
import numpy as np
import torch.nn as nn
import torch
from transformers import AutoModelForSequenceClassification, \
    AutoTokenizer

# specify GPU
device = torch.device("cuda")

# Define path
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
print(dir_path)


"""Load Model"""
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

"""IMPORT"""
list_df=[]

# import the data from Weikmann (2017)
for df in pd.read_csv(dir_path+"/Data/Validation/Weikmans_2017.csv", chunksize=500):

    df['text'] = df['projecttitle']+" "+df['longdescription']

    # initialize function
    preprocess_text = Preprocess_text()

    # Preprocess the texts
    df = preprocess_text.preprocess_text(df, 'text')

    """ Predictions"""

    df['prediction_criterium'] = 0
    df['prediction_criterium'][df.text.str.split().str.len() > 3] = 1

    # define default of climate_relevance -->0
    df['climate_relevance'] = 0
    if df[df['prediction_criterium'] == 1].shape[0] > 0:
        print('predict relevance')
        text_seq, text_mask = tokenize(df.text_preprocessed[df.prediction_criterium == 1].to_list())

        # get predictions for test data
        with torch.no_grad():
            preds = relevance_classifier(text_seq.to(device), text_mask.to(device))
            preds = preds.detach().cpu().numpy()

        pred_relevance = np.argmax(preds, axis=1)

        df['climate_relevance'][df.prediction_criterium == 1] = pred_relevance

    """MULTICLASS"""
    print('number of documents ', df.shape[0])
    print('number of relevant documents ', df[df.climate_relevance == 1].shape[0])
    # mask all texts that are relevant
    df['climate_class_number'] = 500

    if df.text[df.climate_relevance == 1].shape[0] > 0:
        text_seq, text_mask = tokenize(df.text_preprocessed[df.climate_relevance == 1].to_list())

        # get predictions for test data
        with torch.no_grad():
            preds = multiclass(text_seq.to(device), text_mask.to(device))
            preds = preds.detach().cpu().numpy()

        pred_class = np.argmax(preds, axis=1)

        df['climate_class_number'][df.climate_relevance == 1] = pred_class
        df['climate_class'] = df['climate_class_number'].astype(str).replace(label_dict)

    else:
        df['climate_class'] = '500'

    list_df.append(df)
df_total = pd.concat(list_df)
print(df_total)
df_total.to_csv(dir_path+"/Data/Validation/Weikmans_validated.csv", index=False)
print("finished")