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

print(df.shape)
df = df[df.climate_relevance==1]
print('Shape after relevant only:',df.shape)

df=df.rename(columns={'Year':'effective_year','USD_Disbursement':'effective_funding'})

# Add a number for counting the number of projects
df['no_projects'] = 1

#Decriptives over time:
df_time = df
df_time['adaptation_funding'] = 0
df_time['mitigation_funding'] = 0
df_time['environment_funding'] = 0
df_time.adaptation_funding[df_time.meta_category=='Adaptation'] = df_time.effective_funding
df_time.mitigation_funding[df_time.meta_category=='Mitigation'] = df_time.effective_funding
df_time.environment_funding[df_time.meta_category=='Environment'] = df_time.effective_funding
df_time=df_time[['effective_year','adaptation_funding','mitigation_funding','environment_funding']].groupby('effective_year').sum().reset_index()
df_time.to_csv(dir_path+"/Data/Analysis/timeline.csv", index=False)


"""Preprocess descriptives"""
df_2016 = df[df.effective_year>2015]
descriptives_2016 = df_2016[['climate_relevance','climate_class_number','climate_class','meta_category','no_projects','effective_funding']]\
    .groupby(['climate_relevance','climate_class_number','climate_class','meta_category']).sum().reset_index()
descriptives_2016.to_csv(dir_path+"/Data/Analysis/descriptives_2016.csv", index=False)

descriptives_all_years = df[['climate_relevance','climate_class_number','climate_class','meta_category','no_projects','effective_funding']]\
    .groupby(['climate_relevance','climate_class_number','climate_class','meta_category']).sum().reset_index()
descriptives_all_years.to_csv(dir_path+"/Data/Analysis/descriptives_all_years.csv", index=False)
print('FINISHED')
