# consistent and replicable estimation of bilateral climate finance

## Install required packages: 
pip install -r requirements.txt

## Model training: 
-Relevance: Code/Relevance/Relevance_classifier.py

-Multiclass: Code/Multiclass/multi-classifier.py

Weights are stored in Code/saved_weights

## Model prediction
Code/predict_texts/classify_projects.py uses saved weigths from model training to classify new texts from the preprocessed dataframe. Preprocessing steps can be found in Code/Preprocessing/.







