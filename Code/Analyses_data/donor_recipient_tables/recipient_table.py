import pandas as pd
import os
import numpy as np


"""Import"""
# Define path
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

print(dir_path)

df=pd.read_csv(dir_path+"/Data/Analysis/Recipients/recipients_2016.csv")

df = df[df.country_code!='missing']

df['adaptation']=0
df['mitigation']=0
df['climate_finance']=0
df['climate_env']=0

df.adaptation[df.meta_category=='Adaptation']=df.effective_funding
df.mitigation[df.meta_category=='Mitigation']=df.effective_funding
df.climate_finance[df.meta_category.isin(['Adaptation','Mitigation'])]=df.effective_funding
df.climate_env[df.meta_category.isin(['Adaptation','Mitigation','Environment'])]=df.effective_funding

df=df[['RecipientName','climate_finance','climate_env','adaptation','mitigation']].groupby('RecipientName').sum().reset_index()

df=df.sort_values(by='RecipientName')

df.to_csv(dir_path+"/Data/Analysis/Recipients/table_overview.csv")
