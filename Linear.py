
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np


# In[2]:


os.chdir('C:\\Users\\15p001tx\\Desktop\\29th september')
df = pd.read_csv('train.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[10]:


df.isnull().sum()


# In[17]:


new = df.select_dtypes(include=[np.number]).corr()


# In[52]:


correlation_values=new[['SalePrice']]


# In[61]:


# corelation_values[corelation_values[['SalePrice']] < 0.6 | corelation_values[['SalePrice']] > -0.6 ]
selected_features = correlation_values[["SalePrice"]][(correlation_values["SalePrice"]>=0.6)|(correlation_values["SalePrice"]<=-0.6)]


# In[93]:


a = list(selected_features.index)
num = df[a]


# In[94]:


X = num.drop('SalePrice',axis = 1)


# In[95]:


y = num['SalePrice']


# In[96]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts


# In[97]:


X_train, X_test, y_train,y_test = tts(X,y,test_size = 0.3,random_state = 42)


# In[98]:


reg = LinearRegression()


# In[99]:


reg.fit(X,y)


# In[100]:


y_pred = reg.predict(X_test)


# In[101]:


from sklearn.metrics import mean_squared_error


# In[102]:


rmse = np.sqrt(mean_squared_error(y_test,y_pred))


# In[103]:


rmse


# In[104]:


reg.score(X_test,y_test)

