#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

training=pd.read_csv(r'C:\Users\uvans\Downloads\0000000000002429_training_titanic_x_y_train.csv')


# In[2]:


training=pd.read_csv(r'C:\Users\uvans\Downloads\0000000000002429_training_titanic_x_y_train.csv')


# In[3]:


training.head()


# In[5]:


training.shape


# In[6]:


training.info()


# In[7]:


training.isnull().sum()


# In[8]:


# replacing the missing values in "Age" column with mean value
training['Age'].fillna(training['Age'].mean(), inplace=True)


# In[9]:


# finding the mode value of "Embarked" column
print(training['Embarked'].mode())


# In[30]:


# replacing the missing values in "Embarked" column with mode value
training['Embarked'].fillna(training['Embarked'].mode()[0], inplace=True)

# check the number of missing values in each column
training.isnull().sum()
# In[16]:


training.describe()


# In[32]:


training.isnull().sum()


# In[17]:


# finding the number of people survived and not survived
training['Survived'].value_counts()


# In[18]:


training['Sex'].value_counts()


# In[20]:


training['Embarked'].value_counts()


# In[21]:


training.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
training.head()


# In[24]:


X = training.drop(columns = ['Name','Ticket','Survived'],axis=1)
Y = training['Survived']


# In[25]:


X


# In[26]:


Y


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[36]:


print(X.shape, X_train.shape, X_test.shape)


# In[37]:


model = LogisticRegression()
MAIN1=model.fit(X_train, Y_train)


# In[38]:


X_train_prediction = model.predict(X_train)


# In[39]:


print(X_train_prediction)


# In[41]:


from sklearn.metrics import accuracy_score
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[42]:


# accuracy on test data
X_test_prediction = model.predict(X_test)


# In[43]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[59]:


testing=pd.read_csv(r'C:\Users\uvans\Downloads\logistictestingfinal.csv')
testingpd=pd.DataFrame(testing)
testingpd
Y_prediction=model.predict(testingpd)


# In[60]:


Y_prediction


# In[61]:


testingpd


# In[ ]:




