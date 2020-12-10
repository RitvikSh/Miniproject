#!/usr/bin/env python
# coding: utf-8

# # IMPORTING LIBRARIES

# In[138]:


import pandas as pd           ## FOR DATASET RELATED OPERATIONS
import matplotlib.pyplot as plt    ## FOR DATA VISUALIZATION
import seaborn as sns              ## FOR DATA VISUALIZAION
from sklearn.model_selection import train_test_split            ## FOR SPLITTING DATASET INTO TRAINING AND TESTING SET
from sklearn.linear_model import LinearRegression          ## ALGORITHM FOR TRAINING MODEL
from sklearn.tree import DecisionTreeRegressor             ## ALGORITH FOR TRAINING MODEL
from sklearn.metrics import accuracy_score                   ## FOR MODEL ACCURACY


# # IMPORTING DATASET

# In[139]:


dataset  = pd.read_csv('USA_Housing.csv')


# # VIEWING DATA

# In[140]:


dataset.head() 


# In[141]:


dataset['Avg. Area Number of Rooms'] = dataset['Avg. Area Number of Rooms'].astype(int)


# In[142]:


dataset.head() 


# # VIEWING INFO. ABOUT DATASET

# In[143]:


dataset.info() 


# In[144]:


dataset.corr()


# # VISUALIZING DATA

# In[145]:


plt.figure(figsize=(10, 6))
plt.title('Distribution of different prices of houses')
sns.countplot(dataset['Avg. Area Number of Rooms'])                                      
sns.set()
dataset['Avg. Area Number of Rooms'].value_counts()


# In[146]:


dataset.columns


# In[147]:


y = dataset['Price']


# In[148]:


X = dataset[ ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population'] ]


# # SPLITTING THE DATA

# In[149]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# # TRAINING AND TESTING

# In[150]:


model1 = LinearRegression()


# In[151]:


model1.fit(X_train , y_train)


# In[152]:


_pred = model1.predict(X_test)


# In[153]:


model1.coef_


# In[154]:


model1.intercept_


# # CHECKING ACCURACY

# In[155]:


Accuracy = model1.score(X_test, y_test)
print(Accuracy*100, '%')


# # USING ANOTHER MODEL

# In[156]:


model = DecisionTreeRegressor(random_state=0)
model.fit(X_train, y_train)


# In[157]:


Accuracy = model.score(A_test, b_test)
print(Accuracy*100, '%')


# In[ ]:




