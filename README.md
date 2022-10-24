# Consistent and replicable estimation of bilateral climate finance
Estimating the relevance of development assistance projects for climate finance based on textual project descriptions via ClimateFinanceBERT. 
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
Data used for analyses presented in the paper can be found in the [Data directory](/Data/). Climate finance categorizations represent outputs from classifying project descriptions with ClimateFinanceBERT (see Methods in the paper). The raw data from the OECD including the project descriptions can be retrieved from [here](https://stats.oecd.org/DownloadFiles.aspx?DatasetCode=CRS1). Note that all non-English texts must be translated to English (e.g., via GoogleTranslate).   

1. train_set.csv: Annotated training dataset used to train the relevance classifier and the multilabel classifier. Details about the labels are provided in the Methods of the paper. 

2. timeline.csv: Aggregated USD disbursements (in million USD) for all projects and contributors/recipients grouped by year. 

3. descriptives_all_years: Descriptives stats by climate finance category summed up from 2000-2019.

4. descriptives_2016: Descriptives stats by climate finance category summed up from 2016-2019 (see Supplementary Table SI. 2 in the paper). 

### Donors: 
All data used for analysing climate finance provided by contributing countries (all disbursements are provided in million USD).

### Recipients: 
All data used for analysing climate finance received by contributing countries (all disbursements are provided in million USD).

### Rio_markers: 
All data used for comparing classifications from ClimateFinanceBERT with the Rio Marker annotations. 

### Validation: 
Outcomes from comparison between annotations by Weikmans et al., Rio Markers, and ClimateFinanceBERT. 

### User_study: 
Results from the user study comparing annotations by Rio Markers and ClimateFinanceBERT to evaluations of human respondents. 

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

## [Data analysis](Code/Analyses_data/): 
Create the datasets (in the Data folder) to analyse climate finance flows based on classifications by ClimateFinanceBERT (form Model prediction). 

### Postprocessing (Code/Analyses_data/Postprocess_after_classification):
Postprocess the long dataframe (on project level) for subsequent data analyses by grouping data by contributor, recipient, climate finanance categories, years, etc. 

#### Postprocess full data
Creating the meta-categories (Adaptation, Mitigation, Environment) from the output from the model prediction (classify_projects.py).
```shell
python Code/Analysis_data/Postprocess_after_classification/Postprocess_full_data.py
# overwrites output dataframe climate_finance_total.csv
```

#### Postprocess descriptives, recipients, donors:
Grouping the data from long output dataframe (dataframe climate_finance_total.csv) based on targeted analysis.  
```shell
python Code/Analysis_data/Postprocess_after_classification/postprocess_descriptives.py # creates general descriptives dataframes in the data folder (Data/)
python Code/Analysis_data/Postprocess_after_classification/postprocess_recicipients.py # creates data for analysing recipients in thefolder Data/Recipients/
python Code/Analysis_data/Postprocess_after_classification/postprocess_donors.py # creates data for analysing contributors in thefolder Data/Donors/
```

## [Plots](Code/Plots/)
Scripts to plot main figures (Fig. 1 & Fig.2 in the paper) as Jupyter markdown files. 

- Fig1_rio_lines.ipynb: Plot all subfigures of Fig. 1. 
- Fig2_scatter.ipynb: Plot scatterplots from Fig. 2 (Fig. 2 B & C).
- woldmap.ipynb: Plot choropleth map to illustrate funding provided by contributing countries and received by recipient countries in a worldmap. 

## [Rio marker comparison](Code/Rio_markers/)
Scripts for comparisons between ClimateFinanceBERT data and the Rio Marker annotations (from Supplementary Discussion 2 in the paper). 

To create Supplementary Table SI. 3 & 4: 
```shell
python Code/Rio_markers/analyze_shares_rio.py
# creates output dataframes used for Supplemantary Table SI. 3 & 4 (/Data/Analysis/Rio_markers/Precision_aggregated.csv & /Data/Analysis/Rio_markers/Recall_aggregated.csv)
```

- [heatmap.ipynb](Code/Rio_markers/heatmap.ipynb): Jupyter markdown script to plot the heatmaps visualizing Recall and Precision between ClimateFinanceBERT and the Rio Marker annotations on contributing country level.  

## User study (Code/Validation/)
Comparison between annotations for adaptation projects by Weikmans et al and ClimateFinanceBERT (see Methods in the paper). 

To classify project descriptions from Weikmans et al via ClimateFinanceBERT:
```shell
python Code/Validation/Adaptation_Weikmans.py
# creates output dataframe Data/Validation/Weikmans_validated.csv
```

- [Evaluate_outcome.ipynb](Code/Rio_markers/Evaluate_outcome.ipynb): Jupyter markdown to create output table (Supplementary Table SI. 12 in the paper) based on the output from the script above (Weikmans_validated.csv).


## [User study](Code/User_Study/)
User study to compare classifications by ClimateFinanceBERT and annotations by the Rio Markers with results from a user study with human respondents. 

To create Extended Data Fig. 8:
```shell
python Code/User_Study/plot_user_study_results.py
```








