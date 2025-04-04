#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler 
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
import pickle


# In[2]:


df1 = pd.read_csv('loan.csv')


# In[3]:


X1 = df1.drop(["Risk_Flag"], axis=1)
y1 = df1["Risk_Flag"]


# In[4]:


sampler = RandomOverSampler(random_state=42, sampling_strategy=0.45)
X_sampled, y_sampled = sampler.fit_resample(X1, y1)


# In[5]:


df = pd.concat([X_sampled,y_sampled], axis=1)


# In[6]:


X = df.drop("Risk_Flag", axis=1)
y = df["Risk_Flag"]


# In[7]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42)


# In[8]:


rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
rfc.predict(X_test)


# In[9]:


pickle.dump((rfc), open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

