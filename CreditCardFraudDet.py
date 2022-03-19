#!/usr/bin/env python
# coding: utf-8




import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

#pd.set_option('display.max_columns', None)
card = pd.read_csv(r'C:\Users\DELL\Downloads\archives\data\creditcard.csv')
card.head()
card_d=card.copy()
card_d.drop_duplicates(subset=None, inplace=True)
card.shape
card_d.shape

## Assigning removed duplicate datase to original 
card=card_d
card.shape
del card_d

card.Class.value_counts()

#Dropping Time 
estimators=[ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

X1 = card[estimators]
y = card['Class']


# In[30]:


col=X1.columns[:-1]
col

X = sm.add_constant(X1)
reg_logit = sm.Logit(y,X)
results_logit = reg_logit.fit()

def back_feature_elem (data_frame,dep_var,col_list):
   
    while len(col_list)>0 :
        model=sm.Logit(dep_var,data_frame[col_list])
        result=model.fit(disp=0)
        largest_pvalue=round(result.pvalues,3).nlargest(1)
        if largest_pvalue[0]<(0.0001):
            return result
            break
        else:
            col_list=col_list.drop(largest_pvalue.index)

result=back_feature_elem(X,card.Class,col)

new_features=card[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V20','V21', 'V22', 'V23', 'V25', 'V26', 'V27','Class']]
x=new_features.iloc[:,:-1]
y=new_features.iloc[:,-1]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,stratify=y,random_state=42)


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import f1_score

f1_score(y_test, y_pred)

import pickle

with open('./model.pkl', 'wb') as model_pkl:
    pickle.dump(model, model_pkl)

with open('./model.pkl', 'rb') as model_pkl:
    model = pickle.load(model_pkl)

