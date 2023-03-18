#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

df = pd.read_csv(
    '/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
y = pd.read_csv(
    '/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df.head(5)


# In[2]:


# Sort by nulls in columns

pd.set_option('display.max_rows', df.shape[0])
pd.DataFrame(df.isnull().sum().sort_values(ascending = False))


# In[3]:


# Delete columns with a lot of nulls (over50%)

df.drop(columns=['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities', 'FireplaceQu'], 
        inplace = True)
a = df.columns[df.isnull().any()]

# In another columns replace nulls - mode

for i in a:
    df[i] = df[i].fillna(df[i].mode()[0])  
    
df.head(10)


# In[4]:


# Extract SalePrice feature as target array

y = df['SalePrice']
del df['SalePrice']


# In[5]:


# Transfom objects in coloumns to int64

a = df.select_dtypes(include = object)

for i in a:
    label_encoder = preprocessing.LabelEncoder()
    df[i] = label_encoder.fit_transform(df[i])
    df.drop(columns = [], inplace = True)
    
df.head(10)


# In[6]:


# Train/test splitting

x_train, x_test, y_train, y_test = train_test_split(
    df, y, test_size = 0.2, random_state = 1337)


# In[7]:


#Building Model! LGBM Regressor is my favorite regressor for high-dim

lgbm = LGBMRegressor(objective = 'regression', 
                       num_leaves = 13,
                       learning_rate = 0.034428, 
                       n_estimators = 4235,
                       random_state = 1337)

# Fit'n'show rmse

lgbm.fit(x_train, y_train)
lgbm_train_predict = lgbm.predict(x_train)
rmse = np.sqrt(mean_squared_error(y_train, lgbm_train_predict))

print(rmse)


# In[8]:


# Prediction on specific example

pred_0 = lgbm.predict(df)

a = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
y_0 = a['SalePrice']

abs(pred_0[1337] / y_0[1337])


# In[9]:


# Preparing test data

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

pd.set_option('display.max_rows', test.shape[0])
pd.DataFrame(test.isnull().sum().sort_values(ascending = False))


# In[10]:


# Convet test data as train

test.drop(columns=[
    'Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Utilities', 'FireplaceQu'], 
          inplace = True)

a = test.columns[test.isnull().any()]

for i in a:
    test[i] = test[i].fillna(test[i].mode()[0]) 
    
label_encoder = preprocessing.LabelEncoder()

a = test.select_dtypes(include = object)

for i in a:
    label_encoder = preprocessing.LabelEncoder()
    test[i] = label_encoder.fit_transform(test[i])
    test.drop(columns = [], inplace = True)
    
test.head(10)


# In[11]:


# Make some predictions...!

sub = lgbm.predict(test)
sub = pd.DataFrame(sub)


# In[12]:


#Write to csv

submission = pd.read_csv(
    '/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
submission = submission['Id']
submission = pd.DataFrame(submission)
submission['SalePrice'] = sub

submission.to_csv('/kaggle/working/submission.csv', index = False)


# In[13]:


submission = pd.read_csv('/kaggle/working/submission.csv')
submission.head()

