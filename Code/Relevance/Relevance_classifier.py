import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast, RobertaTokenizer, AutoModelForSequenceClassification, \
    AutoTokenizer, TrainingArguments
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from sklearn.utils.class_weight import compute_class_weight
import os
# importing time module
import time

"""This Classifier aims to classify the relevance of paragraphs for the discourse project. 
Paragraphs are labeled as relevant when: 
    1. They address cleantech
    2. They address climate issues
    3. They address finance related to energy projects/ potentially related to energy projects"""


"""Arguments"""
class_number = 2
base_model = 'climatebert/distilroberta-base-climate-f'  #'bert-base-multilingual-uncased'
n_words = 150 #200
batch_size = 64
random_states = 2022
learning_rate = 2e-5

# Define Training args
training_args = {
    "learning_rate":learning_rate,
    "per_device_train_batch_size":batch_size,
    "per_device_eval_batch_size":batch_size,
    "num_train_epochs":50,
    "weight_decay":0.01,
    "n_words": n_words,
    "class_number":class_number
}

"""Import"""
# Define path
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# specify GPU
device = torch.device("cuda")

# Load Data
path = dir_path+ "/Data/train_set.csv"
df = pd.read_csv(path)
df_output=df.copy()

df=df.rename(columns={'label':'class'})
### decision about label
df['label'] = df.relevance
print(df.label)

# split train dataset into train, validation and test sets
train_text, temp_text, train_labels, temp_labels = train_test_split(df['text'], df['label'],
                                                                    random_state=random_states,
                                                                    test_size=0.15,
                                                                    stratify=df['label'])
print(train_text.shape)

val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=random_states,
                                                                test_size=0.5,
                                                                stratify=temp_labels)



"""MODEL """
# import BERT-base pretrained model
auto_model = AutoModelForSequenceClassification.from_pretrained(base_model,
                                                              num_labels=class_number,
                                                                )

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)



"""DECISION ON PADDING LENGTH"""
# tokenize and encode sequences in the training set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = n_words,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = n_words,
    pad_to_max_length=True,
    truncation=True
)

# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = n_words,
    pad_to_max_length=True,
    truncation=True
)


## convert lists to tensors

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())



# wrap tensors
train_data = TensorDataset(train_seq, train_mask, train_y)

# sampler for sampling the data during training
train_sampler = RandomSampler(train_data)

# dataLoader for train set
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# wrap tensors
val_data = TensorDataset(val_seq, val_mask, val_y)

# sampler for sampling the data during training
val_sampler = SequentialSampler(val_data)

# dataLoader for validation set
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)


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



# pass the pre-trained BERT to our define architecture
model = BERT_Arch(auto_model)

# push the model to GPU
model = model.to(device)

"""Training Setup"""
# define the optimizer
optimizer = AdamW(model.parameters(),
                  lr= learning_rate, #2e-5
                  #eps=1e-8, #1e-8
                  weight_decay=training_args["weight_decay"])



#compute the class weights
class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)

print("Class Weights:",class_weights)


# converting list of class weights to a tensor
weights= torch.tensor(class_weights,dtype=torch.float)

# push to GPU
weights = weights.to(device)

# define the loss function
cross_entropy  = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float, device="cuda"))


# function to train the model
def train():

  model.train()

  total_loss, total_accuracy = 0, 0

  # empty list to save model predictions
  total_preds=[]

  # iterate over batches
  for step,batch in enumerate(train_dataloader):

    # progress update after every 50 batches.
    if step % 50 == 0 and not step == 0:
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

    # push the batch to gpu
    batch = [r.to(device) for r in batch]

    sent_id, mask, labels = batch

    # clear previously calculated gradients
    model.zero_grad()

    # get model predictions for the current batch
    preds = model(sent_id, mask)

    # compute the loss between actual and predicted values
    loss = cross_entropy(preds, labels)

    # add on to the total loss
    total_loss = total_loss + loss.item()

    # backward pass to calculate the gradients
    loss.backward()

    # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # update parameters
    optimizer.step()

    # model predictions are stored on GPU. So, push it to CPU
    preds=preds.detach().cpu().numpy()

    # append the model predictions
    total_preds.append(preds)

  # compute the training loss of the epoch
  avg_loss = total_loss / len(train_dataloader)

  # predictions are in the form of (no. of batches, size of batch, no. of classes).
  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  #returns the loss and predictions
  return avg_loss, total_preds



# function for evaluating the model
def evaluate():

  print("\nEvaluating...")

  # deactivate dropout layers
  model.eval()

  total_loss, total_accuracy = 0, 0

  # empty list to save the model predictions
  total_preds = []

  # iterate over batches
  for step,batch in enumerate(val_dataloader):

    # Progress update every 50 batches.
    if step % 50 == 0 and not step == 0:

      # Calculate elapsed time in minutes.
      #elapsed = format_time(time.time() - t0)

      # Report progress.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    # push the batch to gpu
    batch = [t.to(device) for t in batch]

    sent_id, mask, labels = batch

    # deactivate autograd
    with torch.no_grad():

      # model predictions
      preds = model(sent_id, mask)

      # compute the validation loss between actual and predicted values
      loss = cross_entropy(preds,labels)

      total_loss = total_loss + loss.item()

      preds = preds.detach().cpu().numpy()

      total_preds.append(preds)

  # compute the validation loss of the epoch
  avg_loss = total_loss / len(val_dataloader)

  # reshape the predictions in form of (number of samples, no. of classes)
  total_preds  = np.concatenate(total_preds, axis=0)

  return avg_loss, total_preds


# set initial loss to infinite
best_valid_loss = float('inf')

# empty lists to store training and validation loss of each epoch
train_losses=[]
valid_losses=[]

#for each epoch
for epoch in range(training_args["num_train_epochs"]):

    print('\n Epoch {:} / {:}'.format(epoch + 1, training_args["num_train_epochs"]))

    #train model
    train_loss, _ = train()

    #evaluate model
    valid_loss, _ = evaluate()

    #save the best model
    if valid_loss < best_valid_loss:
        print('best')
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), dir_path+'/Code/saved_models/saved_weights_relevance.pt')

    # append training and validation loss
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')

#load weights of best model
path = dir_path+'/Code/saved_models/saved_weights_relevance.pt'
model.load_state_dict(torch.load(path))

# get predictions for test data
with torch.no_grad():
  preds = model(test_seq.to(device), test_mask.to(device))
  preds = preds.detach().cpu().numpy()


preds = np.argmax(preds, axis = 1)
prints_true = list(test_y)
prints_pred = list(preds)
false_preds = []
underlying_texts = []
for i, prediction in enumerate(preds):
    false_preds.append(prediction)
    underlying_texts.append(test_text.tolist()[i])

# Print classification report on test set
print(classification_report(test_y, preds))
