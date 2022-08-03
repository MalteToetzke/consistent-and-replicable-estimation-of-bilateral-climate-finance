import os
import pandas as pd

# Define path
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
print(dir_path)

def csv_import(name, delimiter="|"):
    x = pd.read_csv(name, encoding='utf8', low_memory=False, delimiter=delimiter,
                    dtype={'text': str,
                           "USD_Disbursement": float
                           }
                    )
    return x


# import aid data set
df = csv_import(dir_path+'/Data/df_preprocessed.csv')
print(df.shape)

# only include texts with >8 words
df = df[df.text.str.split().str.len()>8]


"""TEST ANNOTATIONS"""
# by Rio Marker
df_annotation = []
#include Principal and Significant Markers
categories = [1,2]

for category in categories:
    df_annotation.append(df['text'][df.ClimateMitigation == category].sample(1000))
    df_annotation.append(df['text'][df.ClimateAdaptation == category].sample(1000))

# by purpose code
categories = [23210,23110,23250,23220,23510,41010,23230,23240,23270,23610,23630,23360,23340,23330]
for category in categories:
    try:
        df_annotation.append(df['text'][df.PurposeCode == category].sample(20))
    except:
        df_annotation.append(df['text'][df.PurposeCode == category])

#concatenate list of stratefied samples
df_annotation_set = pd.concat(df_annotation)
#shuffle to random order
df_annotation_set = df_annotation_set.sample(frac=1).reset_index(drop=True).shuffle()
df_annotation_set.to_csv(dir_path+'/Data/annotation_set.csv', encoding='utf8', index=False, header=True, sep='|')