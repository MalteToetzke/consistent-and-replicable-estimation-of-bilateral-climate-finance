import pandas as pd
import os
import numpy as np


"""Import"""
# Define path
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

print(dir_path)


def csv_import(name, delimiter="|"):
    x = pd.read_csv(name, encoding='utf8', low_memory=False, delimiter=delimiter,
                    dtype={'text': str,
                           "USD_Disbursement": float
                           }
                    )
    return x

df = csv_import(dir_path+'/Data/climate_finance_total.csv')

df=df[df.Year>2015]
df=df.rename(columns={'USD_Disbursement':'effective_funding'})

df['adaptation']=0
df['mitigation']=0
df['climate_finance']=0
df['climate_env']=0

df['rio_clim_prin']=0
df['rio_adap_prin']=0
df['rio_miti_prin']=0
df['rio_clim_all']=0
df['rio_adap_all']=0
df['rio_miti_all']=0

df.adaptation[df.meta_category=='Adaptation']=df.effective_funding
df.mitigation[df.meta_category=='Mitigation']=df.effective_funding
df.climate_finance[df.meta_category.isin(['Adaptation','Mitigation'])]=df.effective_funding
df.climate_env[df.meta_category.isin(['Adaptation','Mitigation','Environment'])]=df.effective_funding

df.rio_clim_prin[(df.ClimateMitigation==2)|(df.ClimateAdaptation==2)]=df.effective_funding
df.rio_adap_prin[(df.ClimateAdaptation==2)]=df.effective_funding
df.rio_miti_prin[(df.ClimateMitigation==2)]=df.effective_funding
df.rio_clim_all[(df.ClimateMitigation.isin([2,1]))|(df.ClimateAdaptation.isin([2,1]))]=df.effective_funding
df.rio_adap_all[(df.ClimateAdaptation.isin([2,1]))]=df.effective_funding
df.rio_miti_all[(df.ClimateMitigation.isin([2,1]))]=df.effective_funding

print("SUM CLimate finance")
print(df.climate_finance.sum())
df=df[['DonorName','climate_finance','climate_env','adaptation','mitigation','rio_clim_prin',
'rio_adap_prin','rio_miti_prin','rio_clim_all','rio_adap_all','rio_miti_all'
       ]].groupby('DonorName').sum().reset_index()

df=df.sort_values(by='DonorName')
df = df[df.climate_finance!=0]
print("sum kuwait, etc.")
print(df.climate_finance[df.DonorName.isin(['Cyprus','Croatia','Kuwait','Saudi Arabia','Turkey'])].sum())
df = df[df.DonorName.isin(['Cyprus','Croatia','Kuwait','Saudi Arabia','Turkey'])==False]

#df.to_csv(dir_path+"/Data/Analysis/Donors/table_overview.csv")
