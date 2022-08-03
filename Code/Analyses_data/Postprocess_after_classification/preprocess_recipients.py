import pandas as pd
import os
import numpy as np


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

print(df.shape)
df = df[df.climate_relevance==1]
print('Shape after relevant only:',df.shape)

df=df.rename(columns={'Year':'effective_year','USD_Disbursement':'effective_funding'})

## Country codes
codes = pd.read_csv(dir_path+'/Data/Analysis/Country_codes.csv', encoding='latin-1')

# Incomegroups
Incomegroup = pd.read_csv(dir_path+"/Data/Analysis/IncomeGroup.csv")

# Add a number for counting the number of projects
df['no_projects'] = 1


def preprocess_bubble(df, dim1, dim2,funding_type):  # dim differs between donor or recipient? or just country code

    df = df[['meta_category', dim1, dim2, 'effective_funding', 'country_code','gdp']].groupby(['meta_category',
                                                                                   dim1, dim2, 'country_code','gdp']
                                                                                  , observed=True) \
        .sum().reset_index()  ###,observed=True?

    print(df[df.country_code=='missing'])
    df=df[df.country_code!='missing']
    funding_name = funding_type+'_funding'
    df[funding_name] = 0
    df[funding_name][df.meta_category==funding_type] = df['effective_funding']
    print('test if climate works', df)
    df = df[[dim1, dim2, funding_name,'effective_funding', 'country_code','gdp']].groupby([dim1, dim2, 'country_code','gdp']
                                                                                          ,
                                                                                          observed=True).sum().reset_index()

    perc_name = 'perc_'+funding_type
    df[perc_name] = df[funding_name] / (df[funding_name].sum()) * 100
    print('Plausibility')
    print(df[perc_name].sum())

    return df

def add_Country_codes(df, codes):
    # adding country_codes to df
    df['country_code'] = ''

    country = codes['Recipient'].tolist()
    for i, row in df.iterrows():
        country_name = row['RecipientName']
        if row['RecipientName'] in country:

            df.at[i, 'country_code'] = \
                codes[codes['Recipient'] == country_name].reset_index(drop=True)['Code'][0]

        else:
            df.at[i, 'country_code'] = 'missing'
            # print(row['RecipientName'])

    df['country_code'][df['RecipientName'].str.contains('Ivoire', na=False, regex=True)] = 'CIV'
    df['country_code'][df['RecipientName'] == 'Anguilla'] = 'AIA'
    df['country_code'][df['RecipientName'] == 'Malta'] = 'MLT'
    df['country_code'][df['RecipientName'] == 'Slovenia'] = 'SVN'
    return df

def add_Incomegroup(df, codes):
    # adding country_codes to df
    df['IncomegroupName'] = ''

    country = codes['RecipientName'].tolist()
    for i, row in df.iterrows():
        country_name = row['RecipientName']
        if row['RecipientName'] in country:

            df.at[i, 'IncomegroupName'] = \
                codes[codes['RecipientName'] == country_name].reset_index(drop=True)['IncomegroupName'][0]

        else:
            df.at[i, 'IncomegroupName'] = 'missing'
            # print(row['RecipientName'])

    return df




"""Preprocess Recipients:"""
print("Recipients:")

df = df[['effective_year','RecipientName','climate_class_number','climate_class','meta_category','no_projects','effective_funding']]\
    .groupby(['effective_year','RecipientName','climate_class_number','climate_class','meta_category']).sum().reset_index()

df = add_Country_codes(df, codes)

df = add_Incomegroup(df, Incomegroup)
df.IncomegroupName[df.country_code=='CIV']='LMICs'
print(df[df.IncomegroupName == 'missing'])

small_island_developing_states=pd.read_csv(dir_path+'/Data/Analysis/SMDS_isos.csv',encoding='latin-1')

small_island_developing_states=small_island_developing_states['ISO']
df['country_type']=df['IncomegroupName']
df['country_type']=df['country_type'].astype(str)
df['country_type'][df.country_code.isin(list(small_island_developing_states))]='Small Island Developing States'
df['country_type'][df.country_type=='Part I unallocated by income']='Region'
df['country_type'][df.country_type=='Other LICs']='LMICs'

### Add gdp
gdp = pd.read_csv(dir_path+'/Data/Analysis/gdp.csv')
df['gdp'] = np.nan

country = list(gdp['Country_Code'])
for i, row in df.iterrows():
    country_name = row['country_code']
    if row['country_code'] in country:

        df.at[i, 'gdp'] = \
            gdp[gdp['Country_Code'] == country_name].reset_index(drop=True)['gdp_2015_2019'][0]

    else:
        # df2.at[i, 'pop'] = 'missing'
        print(row['RecipientName'])

df['gdp'][df.country_code == 'COK'] = 355000000
df['gdp'][df.country_code == 'NIU'] = 24000000
df['gdp'][df.country_code == 'MSR'] = 67000000
df['gdp'][df.country_code == 'SHN'] = 39000000
df['gdp'][df.country_code == 'TKL'] = 10000000
df['gdp'][df.country_code == 'AIA'] = 313000000
df['gdp'][df.country_code == 'WLF'] = 171000000

# save grouped output
df.to_csv(dir_path+"/Data/Analysis/recipients.csv", index=False)

# only 2016-2019 and all merged
###Edit after revision: Use 2000-2009; 2010-2012; 2013-
df2 = df[df.effective_year>2015]
df2 = df2[['RecipientName','country_code','country_type','gdp','climate_class_number','climate_class','meta_category','no_projects','effective_funding']]\
    .groupby(['RecipientName','country_code','country_type','gdp','climate_class_number','climate_class','meta_category']).sum().reset_index()

#inner scope of climate funding
df2['climate_funding'] = 0
df2['climate_funding'][(df2.meta_category.isin(["Adaptation","Mitigation"]))] = df2.effective_funding
# save 2016 output used for Worldmap
df2.to_csv(dir_path+"/Data/Analysis/recipients_2016.csv", index=False)


## Preprocess data for the bubble diagram
rec_miti =preprocess_bubble(df2,'RecipientName','country_type',funding_type='Mitigation')
rec_miti.to_csv(dir_path+"/Data/Analysis/recipients_miti_16_19.csv", encoding='utf8',header=True,index=False)

rec_adap =preprocess_bubble(df2,'RecipientName','country_type',funding_type='Adaptation')
rec_adap.to_csv(dir_path+"/Data/Analysis/recipients_adap_16_19.csv", encoding='utf8',header=True,index=False)


## Get descriptives over all recipients
df2['climate finance (including nature)'] = df2.effective_funding
df2['adaptation_funding'] = 0
df2['mitigation_funding'] = 0
df2['environment_funding'] = 0
df2.adaptation_funding[df2.meta_category=='Adaptation'] = df2.effective_funding
df2.mitigation_funding[df2.meta_category=='Mitigation'] = df2.effective_funding
df2.environment_funding[df2.meta_category=='Environment'] = df2.effective_funding

recipients_descriptives = df2[['RecipientName','climate_funding','climate finance (including nature)','environment_funding','mitigation_funding','adaptation_funding','effective_funding']]\
    .groupby(['RecipientName','climate_funding','climate finance (including nature)','environment_funding','mitigation_funding','adaptation_funding']).sum().reset_index()

recipients_descriptives.to_csv(dir_path+"/Data/Analysis/recipients_descriptives_2016.csv", index=False)