""" THis is to add meta-categories to the files and change the category names"""

import pandas as pd
import os

"""Import"""
# Define path
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

print(dir_path)

def csv_import(name, delimiter="|"):
    x = pd.read_csv(name, encoding='utf8', low_memory=False, delimiter=delimiter,
                    dtype={'text': str,
                           "USD_Disbursement": float
                           }
                    )
    return x

df = csv_import(dir_path+'/Data/climate_finance_total.csv')

length_start = df.shape[0]
if length_start<2700000:
    print(length_start)
    quit()

## change some names
df.climate_class[df.climate_class=="Biofuel-energy"]="Bioenergy"
df.climate_class[df.climate_class=="Other-environment-protection-projects"]="Other-environment-projects"

## add meta categories
adaptation_categories = [0]
environment_categories = [4, 6, 7, 8]
mitigation_categories = [1,2,3,5,9,10,11,12,13]

df['meta_category']='None'
df['meta_category'][df.climate_class_number.isin(adaptation_categories)] = 'Adaptation'
print('Adaptation',df[df.climate_class_number.isin(adaptation_categories)].shape[0])
df['meta_category'][df.climate_class_number.isin(mitigation_categories)] = 'Mitigation'
print('Mitigation',df[df.climate_class_number.isin(mitigation_categories)].shape[0])
df['meta_category'][df.climate_class_number.isin(environment_categories)] = 'Environment'
print('Environment',df[df.climate_class_number.isin(environment_categories)].shape[0])

if df[df.meta_category=='None'].shape[0]==df[df.climate_relevance==0].shape[0]:
    print('Plausibility passed')

else:
    print('META SHAPE NONE: ',df[df.meta_category=='None'].shape[0])
    print('Relevance SHAPE 0: ', df[df.relevance==0].shape[0])
    quit()

length_end = df.shape[0]

if length_end==length_start:
    print("Second test passed")

else:
    print("Start Shape: ",length_start)
    print("End Shape: ", length_end)
    quit()


df.to_csv(dir_path+'/Data/climate_finance_total.csv', encoding='utf8', index=False, header=True, sep='|')
