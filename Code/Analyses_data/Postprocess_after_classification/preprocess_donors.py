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

# rename columns
df=df.rename(columns={'Year':'effective_year','USD_Disbursement':'effective_funding'})

# Add a number for counting the number of projects
df['no_projects'] = 1

## Country codes
donors = pd.read_csv(dir_path+'/Data/Analysis/DonorNames.csv')


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


"""Preprocess"""
df = df[['effective_year','DonorName','climate_class_number','climate_class','meta_category','no_projects','effective_funding']]\
    .groupby(['effective_year','DonorName','climate_class_number','climate_class','meta_category']).sum().reset_index()


df['country_code'] = ''

country = list(donors['Name'])
for i, row in df.iterrows():
    country_name = row['DonorName']
    if row['DonorName'] in country:

        df.at[i, 'country_code'] = \
            donors[donors['Name'] == country_name].reset_index(drop=True)['country_code'][0]

    else:
        df.at[i, 'country_code'] = 'missing'
        # print(row['RecipientName'])


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
        print(row['DonorName'])





# save grouped output
df.to_csv(dir_path+"/Data/Analysis/donors.csv", index=False)

# only 2016-2019 and all merged
df2 = df[df.effective_year>2015]
df2 = df2[['DonorName','country_code','gdp','climate_class_number','climate_class','meta_category','no_projects','effective_funding']]\
    .groupby(['DonorName','country_code','gdp','climate_class_number','climate_class','meta_category']).sum().reset_index()

#inner scope of climate funding
df2['climate_funding'] = 0
df2['climate_funding'][(df2.meta_category.isin(["Adaptation","Mitigation"]))] = df2.effective_funding
# save 2016 output used for Worldmap
df2.to_csv(dir_path+"/Data/Analysis/donors_2016.csv", index=False)

df2['DonorType']='bilateral donor'

## Preprocess data for the bubble diagram
rec_miti =preprocess_bubble(df2,'DonorName','DonorType',funding_type='Mitigation')
rec_miti.to_csv(dir_path+"/Data/Analysis/donors_miti_16_19.csv", encoding='utf8',header=True,index=False)

rec_adap =preprocess_bubble(df2,'DonorName','DonorType',funding_type='Adaptation')
rec_adap.to_csv(dir_path+"/Data/Analysis/donors_adap_16_19.csv", encoding='utf8',header=True,index=False)


## Get descriptives over all donors
df2['climate finance (including nature)'] = df2.effective_funding
df2['adaptation_funding'] = 0
df2['mitigation_funding'] = 0
df2['environment_funding'] = 0
df2.adaptation_funding[df2.meta_category=='Adaptation'] = df2.effective_funding
df2.mitigation_funding[df2.meta_category=='Mitigation'] = df2.effective_funding
df2.environment_funding[df2.meta_category=='Environment'] = df2.effective_funding

donors_descriptives = df2[['DonorName','climate_funding','climate finance (including nature)','environment_funding','mitigation_funding','adaptation_funding','effective_funding']]\
    .groupby(['DonorName','climate_funding','climate finance (including nature)','environment_funding','mitigation_funding','adaptation_funding']).sum().reset_index()

donors_descriptives.to_csv(dir_path+"/Data/Analysis/donors_descriptives_2016.csv", index=False)