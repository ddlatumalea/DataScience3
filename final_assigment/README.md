# Disphonic Voice Pattern Analysis of Patients in Parkinson's Diseasing Using Minimum Interclass Probability Risk Feature Selection and Bagging Ensemble Learning Method.

Dataset: http://archive.ics.uci.edu/ml/datasets/Parkinsons

Research paper: Research Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5434464/

Must haves:
- Load, inspect and clean data.
- Split data into training, validation and test set.
- Use a simple classifier as baseline.
- Train various classifiers.
- Motivate your choice in relation to the characteristics of the dataset.
- Combine classifiers into ensemble learning, that outperforms others.

# Overview
Dysphonia is a disorder that causes difficulties in voice production. It can be observed with the use of vowel sounds. About 70-80% of Idopathic Parkinson's disease (IPD) patients suffer from dysphonia. The quantification of vocal parameters helps in understanding disease progression.

Patients with IDP have significant phonetic variability in comparison with healthy controls.

It is of importance to be able to identify patients that have IDP, to be able to treat it. This can be predicted by a model that learns the vocal parameters.

**Current solutions**:
- Logistic regression
- Support vector machine
- Bagging ensemble

**Problem frame**:

It is a *model-based offline* *supervised* *classification* problem. We have to classify healthy participants and participants with IDP. This requires pattern detection in the training data to build a predictive model. It is a binary classification problem.

**Performance measure**:

The dataset is imbalanced, therefore accuracy may not be the best measure. 

It is important to find patients with IDP. Therefore, we want as much true positives and true negatives as possible. False positives are not as good, but won't pose a big problem, as someone would need to do additional tests. False negatives are detrimental and should not occur. It will cause harm if someone has IDP but it is not predicted.

The ROC-curve is also usefull for imbalanced classes and it provides a visual trade-off between FPR and TPR.

Thus, measures that should be used are:
- Sensitivity (true positive rate)
- Recall/sensitivity (true negative rate)
- ROC-curve 

The performance measures are aligned with the business objective to distinguish between healthy patients and IDP patients.

**Minimum performance**:

Wu et al. achieved a high AUC of 0.9558 using Bagging with ICPR features, and a sensitvity of 0.9796. They used specific feature extraction techniques that contributed to their result. In this project we aim to get AUC and sensitivity of atleast 0.9. 

**Folder**:
- For a thorough exploration of the data, see `src/exploration/exploration.ipynb`.
- For baseline model results, see `src/data/results/` or `src/model_tuning/hyperaparams_tuning.ipynb`.
- For the data splitting see `src/splitters/splitters.py`.

# Steps to create a model
1) run `preprocessing.py` to create the datasets

```python
python preprocessing.py
```

2) run `main.py` to evaluate 8 models on 4 different data representations, totalling 32 models. Models are important, but so is the representation of data. Therefore, multiple representations are used to improve performance.

```python
python main.py
```

3) run `hyperparams_tuning.py` to create an optimized model.

```python
python hyperparams_tuning.py
```

# Conclusions
The best performing model is a `Random Forest`. The model has a recall around 1.0 an accuracy of 0.95 and a roc_auc of 0.9. The recall is competitive with the model from the paper. However, the roc_auc is lower. I find the recall very important, due to the imbalanced data and the importance of classifying patients with Parkinson's disease correctly.

The SMOTE algorithm was also used, but the models did not perform better. An overview of all the results of models can be found in `data/results` or `model_tuning/hyperparams_tuning.ipynb`.

All the results were cross validated, evaluated and the best performing algorithms were optimized (Random Forest). 

In the future it would be helpful to gather multiple data to classify patients. It is clearly visible that patients with Parkinson's disease have a higher variance. Therefore, patients with Parkinson's disease have a phenotype that varies more than the phenotype of healthy people. This is important, as it means that PD patients probably need their own personalized medical treatment. This requires personalized medicine, which take more research effort.
