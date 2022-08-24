import pandas as pd

dir_path = ''


print(dir_path)

def csv_import(name, delimiter="|"):
    x = pd.read_csv(name, encoding='utf8', low_memory=False, delimiter=delimiter,
                    dtype={'text': str,
                           "USD_Disbursement": float
                           }
                    )
    return x


df = csv_import(dir_path+'Data/climate_finance_total.csv')
df = df[df.Year>2015]

conditions_bert = [['Mitigation'],['Adaptation'],['Environment'],
                       ['Mitigation','Adaptation'],
                       ['Mitigation','Adaptation','Environment']]
categories = ['Mitigation','Adaptation','Environment','Climate Finance','Climate Finance including Environment']
overlap_mit_princ=[]
overlap_mit_sign=[]
overlap_ada_princ =[]
overlap_ada_sign =[]
overlap_rio_nan = []
overlap_rio_null = []
overlap_clim_princ=[]
overlap_clim_sign=[]

for column in conditions_bert:
    overlap_mit_princ.append(df[(df.meta_category.isin(column))&(df.ClimateMitigation==2)].shape[0]
                              /df[df.meta_category.isin(column)].shape[0])
    overlap_mit_sign.append(df[(df.meta_category.isin(column))&((df.ClimateMitigation==2)|(df.ClimateMitigation==1))].shape[0]/
                             df[df.meta_category.isin(column)].shape[0])
    overlap_ada_princ.append(df[(df.meta_category.isin(column))&(df.ClimateAdaptation==2)].shape[0]/
                             df[(df.meta_category.isin(column))].shape[0])
    overlap_ada_sign.append(df[(df.meta_category.isin(column))&((df.ClimateAdaptation==2)|(df.ClimateAdaptation==1))].shape[0]/
                             df[(df.meta_category.isin(column))].shape[0])
    overlap_rio_nan.append(df[(df.meta_category.isin(column))&((df.ClimateAdaptation.isnull())&(df.ClimateMitigation.isnull()))].shape[0]/
                             df[(df.meta_category.isin(column))].shape[0])
    overlap_rio_null.append(df[(df.meta_category.isin(column))&((df.ClimateAdaptation==0)&(df.ClimateMitigation==0))].shape[0]/
                             df[(df.meta_category.isin(column))].shape[0])

    overlap_clim_princ.append(df[(df.meta_category.isin(column))&((df.ClimateAdaptation==2)|(df.ClimateMitigation==2))].shape[0]/
                              df[(df.meta_category.isin(column))].shape[0])
    overlap_clim_sign.append(df[(df.meta_category.isin(column))&((df.ClimateAdaptation==2)|(df.ClimateMitigation==2)|(df.ClimateAdaptation==1)|(df.ClimateMitigation==1))].shape[0]/
                             df[(df.meta_category.isin(column))].shape[0])

df_heatmap=pd.DataFrame(data={'ClimateFinanceBERT Categories':categories,
                              'Mitigation Principal (RM)':overlap_mit_princ,
                              'Mitigation Significant (RM)':overlap_mit_sign,
                            'Adaptation Principal (RM)':overlap_ada_princ,
                            'Adaptation Significant (RM)':overlap_ada_sign,
                              'Climate Principal (RM)':overlap_clim_princ,
                            'Climate Significant (RM)':overlap_clim_sign,
                              'Not Evaluated':overlap_rio_nan,
                              'Not relevant':overlap_rio_null
                              })
df_heatmap.to_csv(dir_path+'/Data/Analysis/Rio_markers/Recall_aggregated.csv')


overlap_mit_princ=[]
overlap_mit_sign=[]
overlap_ada_princ =[]
overlap_ada_sign =[]
overlap_clim_princ=[]
overlap_clim_sign=[]

for column in conditions_bert:
    overlap_mit_princ.append(df[(df.meta_category.isin(column))&(df.ClimateMitigation==2)].shape[0]
                              /df[df.ClimateMitigation==2].shape[0])
    overlap_mit_sign.append(df[(df.meta_category.isin(column))&((df.ClimateMitigation==1)|(df.ClimateMitigation==2))].shape[0]
                              /df[((df.ClimateMitigation==1)|(df.ClimateMitigation==2))].shape[0])
    overlap_ada_princ.append(df[(df.meta_category.isin(column))&(df.ClimateAdaptation==2)].shape[0]
                              /df[df.ClimateAdaptation==2].shape[0])
    overlap_ada_sign.append(df[(df.meta_category.isin(column))&((df.ClimateAdaptation==1)|(df.ClimateAdaptation==2))].shape[0]
                              /df[((df.ClimateAdaptation==1)|(df.ClimateAdaptation==2))].shape[0])


    overlap_clim_sign.append(df[(df.meta_category.isin(column))&((df.ClimateAdaptation==1)|(df.ClimateMitigation==1)|(df.ClimateAdaptation==2)|(df.ClimateMitigation==2))].shape[0]
                              /df[((df.ClimateAdaptation==1)|(df.ClimateMitigation==1)|(df.ClimateAdaptation==2)|(df.ClimateMitigation==2))].shape[0])
    overlap_clim_princ.append(df[(df.meta_category.isin(column))&((df.ClimateAdaptation==2)|(df.ClimateMitigation==2))].shape[0]
                              /df[((df.ClimateAdaptation==2)|(df.ClimateMitigation==2))].shape[0])

df_heatmap=pd.DataFrame(data={'ClimateFinanceBERT Categories':categories,
                              'Mitigation Principal (RM)':overlap_mit_princ,
                              'Mitigation Significant (RM)':overlap_mit_sign,
                            'Adaptation Principal (RM)':overlap_ada_princ,
                            'Adaptation Significant (RM)':overlap_ada_sign,
                              'Climate Principal (RM)': overlap_clim_princ,
                              'Climate Significant (RM)': overlap_clim_sign,

                              })

df_heatmap.to_csv(dir_path+'/Data/Analysis/Rio_markers/Precision_aggregated.csv')
