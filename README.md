# Consistent and replicable estimation of bilateral climate finance
Estimating the relevance of development assistance projects for climate finance based on textual project descriptions. 
The repository represents the underlying code for the scientific paper "Consistent and replicable estimation of bilateral climate finance" by Malte Toetzke, Anna Stünzi, Florian Egli. 

# Guideline
How to work with this repository

## Installation
Python Interpreter: Python3.8

Install required packages: 
```shell
pip install -r requirements.txt
```

## Data
Data used for analyses presented in the paper can be found in the "Data" directory. Climate finance categorizations represent outputs from classifying project descriptions with ClimateFinanceBERT (see Methods in the paper). The raw data from the OECD including the project descriptions can be retrieved from https://stats.oecd.org/DownloadFiles.aspx?DatasetCode=CRS1. Note that all non-English texts must be translated to English (e.g., via GoogleTranslate).   

train_set.csv: Annotated training dataset used to train the relevance classifier and the multilabel classifier. Details about the labels are provided in the Methods of the paper. 

timeline.csv: Aggregated USD disbursements (in million USD) for all projects and contributors/recipients grouped by year. 

descriptives_all_years: Descriptives stats by climate finance category summed up from 2000-2019.

descriptives_2016: Descriptives stats by climate finance category summed up from 2016-2019 (see Supplementary Table SI. 2 in the paper). 

###Donors: 
All data used for analysing climate finance provided by contributing countries (all disbursements are provided in million USD).

###Recipients: 
All data used for analysing climate finance received by contributing countries (all disbursements are provided in million USD).


## Usage

## Model training: 
Scipts used to train the relevance classifier and the multilabel classifier of ClimateFinanceBERT. Trained weights are stored in the directory Code/saved_weights/.

### Relevance classifier: 
```shell
python Code/Relevance/Relevance_classifier.py
```

### Multilabel classifier:
```shell
python Code/Multiclass/multi-classifier.py
```

## Model prediction:
Use the saved weigths from model training to classify new project descriptions regarding climate finance relevance and corresponding climate finance category. 

```shell
python Code/predict_texts/classify_projects.py 
# creates output dataframe climate_finance_total.csv
```

## Data analysis (Code/Analyses_data/): 
Create the datasets (in the Data folder) to analyse climate finance flows based on classifications by ClimateFinanceBERT (form Model prediction). 

### Postprocessing (Code/Analyses_data/Postprocess_after_classification):
Postprocess the long dataframe (on project level) for subsequent data analyses by grouping data by contributor, recipient, climate finanance categories, years, etc. 

### Postprocess full data
Creating the meta-categories (Adaptation, Mitigation, Environment) from the output from the model prediction (classify_projects.py).
```shell
python Postprocess_full_data.py
# overwrites output dataframe climate_finance_total.csv
```

### Postprocess descriptives, recipients, donors:
Grouping the data from long output dataframe (dataframe climate_finance_total.csv) based on targeted analysis.  
```shell
python postprocess_descriptives.py # creates general descriptives dataframes in the data folder (Data/)
python postprocess_recicipients.py # creates data for analysing recipients in thefolder Data/Recipients/
python postprocess_donors.py # creates data for analysing contributors in thefolder Data/Donors/
```








