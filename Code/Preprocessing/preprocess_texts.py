import os
import pandas as pd
from helpers import Preprocess_text

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 10)
# Define path
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(dir_path)

"""Import the data in chunks"""

def csv_import(name, delimiter="|", chunksize=10 ** 5):
    x = pd.read_csv(name, encoding='utf8', low_memory=False, delimiter=delimiter, chunksize=chunksize,
                    dtype={'text': str,
                           "USD_Disbursement": float
                           }
                    )
    return x

# initialize function
preprocess_text = Preprocess_text()
#List to append chunks
chunk_list = []

# import dataset with translated project descriptions in chunks
for chunk in csv_import(dir_path+'/Data/df_translated.csv'):
    chunk = chunk[pd.to_numeric(chunk['Year'], errors='coerce').notnull()]
    chunk['text'][chunk.text.isnull()] = ''
    chunk.Year = chunk.Year.astype(int)
    print(chunk.shape)

    if chunk.shape[0]>0:
        # preprocess the texts (lowercase, remove punctuations)
        chunk = preprocess_text.preprocess_text(chunk, 'text')
        chunk['text'] = chunk.text_preprocessed
        chunk.drop('text_preprocessed', axis=1, inplace=True)
    chunk_list.append(chunk)
    del chunk


df = pd.concat(chunk_list)
print(df.shape)

###Check if texts for 2018 are still correct
print(list(df.Year.drop_duplicates()))

df.to_csv(dir_path + '/Data/df_preprocessed.csv', encoding='utf8', index=False, header=True, sep='|')
