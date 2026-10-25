# %% [markdown]
# # 5.5 Train 'churn-model'
# 
# We are using the previous model that was trained for predicting churn and evaluated it. Now, the notebook is for deployment.

# Libraries
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import warnings # remove when use in production or debugging
from sklearn.exceptions import ConvergenceWarning # use for exceptions in compilation
warnings.filterwarnings("ignore", category=ConvergenceWarning) # ignore the log about 'Increase the number of iterations to improve the convergence (max_iter=1000).'


# ### Parameters
C = 1.0     # Control-regularization
n_splits = 5
output_file = f'model_C={C}.bin'


# Cleaning data
df = pd.read_csv('./data/data-week-3.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

# Split the dataframe into train & test
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

# Separete columns that gonna use for train & predict
numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# ## Validation
print(f'-- Validation -- w/C ={C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0
for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)
    print(f'AUC on Fold {fold} is {auc}')
    fold += 1

print('\n::Validation Results::')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

print('-- Training : Final Model --')

dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)

print(f'> AUC ={auc}')

# ## Save the model

with open(output_file, 'wb') as f_out: 
    pickle.dump((dv, model), f_out)

print(f'> The model is saved to {output_file}')

# %%
