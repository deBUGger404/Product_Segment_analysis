#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_csv("Menu Items.csv")


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


## checking null values
df.isna().sum()


# In[7]:


## Information for all columns
df.describe()


# In[8]:


## checking for duplicate rows
df.shape, df.drop_duplicates().shape


# In[9]:


## converting price column into float
df['Price'] = df['Price'].str.replace("$","").astype(float)


# In[10]:


## removing rows where description and item are null
df = df[(~df['Description'].isna()) & (~df['Item'].isna())]


# In[11]:


df.isna().sum()


# In[12]:


df1 = df.groupby('Restaurant').agg({'Price':['max','mean','count']}).reset_index()


# In[13]:


max(df1['Price','mean'])


# In[17]:


df1.hist()


# In[18]:


df['Price'].hist()


#                                                             # End Part1
