#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')


# In[2]:


df = pd.read_csv('data/Axis_Bank.csv')
df.head(6)


# In[3]:


df.shape


# In[5]:


plt.figure(figsize=(16,8))
plt.title('Netflix')
plt.xlabel('Days')
plt.ylabel('Close Price USD ($)')
plt.plot(df['Close'])
plt.show()


# In[6]:


df = df[['Close']]
df.head(4)


# In[9]:


future_days = 25
df['Prediction'] =df[['Close']].shift(-future_days)
df.head(4)


# In[10]:


X = np.array(df.drop(['Prediction'],1))[:-future_days]
print(X)


# In[12]:


y = np.array(df['Prediction'])[:-future_days]
print(y)


# In[13]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25)


# In[14]:


tree = DecisionTreeRegressor().fit(x_train,y_train)
lr = LinearRegression().fit(x_train,y_train)


# In[15]:


x_future = df.drop(['Prediction'],1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
x_future


# In[16]:


tree_prediction = tree.predict(x_future)
print(tree_prediction)
print()
lr_prediction = lr.predict(x_future)
print(lr_prediction)


# In[26]:


prediction = tree_prediction
valid = df[X.shape[0]:]
valid['prediction'] = prediction
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','prediction']])
plt.legend(['Orig','Val','Pred'])
plt.show()


# In[27]:


prediction = lr_prediction
valid = df[X.shape[0]:]
valid['prediction'] = prediction
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close USD ($)')
plt.plot(df['Close'])
plt.plot(valid[['Close','prediction']])
plt.legend(['Orig','Val','Pred'])
plt.show()


# In[ ]:




