#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[2]:


cols = 'id'
df_train= pd.read_csv('C:\\Users\\Devjp\\Downloads\\Housing.csv')


# In[3]:


df_train.columns


# In[4]:


#from sklearn.preprocessing import LabelEncoder
#df_train= pd.read_csv('C:\\Users\\Devjp\\Downloads\\Housing.csv')

# Create a LabelEncoder instance
#label_encoder = LabelEncoder()

# Fit and transform the dataset
#numerical_data = label_encoder.fit_transform(data)

#print(numerical_data)


# In[5]:


df_train.head()


# In[6]:


df_train['furnishingstatus'].unique()


# In[7]:


df_train['mainroad'].unique()


# In[8]:


df_train['guestroom'].unique()


# In[9]:


df_train['basement'].unique()


# In[10]:


df_train['hotwaterheating'].unique()


# In[11]:


df_train['airconditioning'].unique()


# In[12]:


df_train['prefarea'].unique()


# In[13]:


for column in df_train.columns:
    # Use .unique() to get unique values for each feature
    unique_values = df_train[column].unique()

    # Print the column name and its unique values
    print(f"Unique values for {column}: {unique_values}")


# In[14]:


df_train.shape


# In[15]:


df_train.isna().sum()


# In[16]:


df_train.info()


# In[17]:


df_train


# In[18]:


#Not usefull just doing for fun :)
df_train.describe()


# In[19]:


df_train['price'].dtype


# ## Univariate Analysis

# In[20]:


sns.displot(df_train['price'])


# In[21]:


print("Skewness: %f" % df_train['price'].skew())
print("Kurtosis: %f" % df_train['price'].kurt())


# In[22]:


Fare_scaled= StandardScaler().fit_transform(df_train['price'][:,np.newaxis]);
low_range= Fare_scaled[Fare_scaled[:,0].argsort()][:10]
high_range= Fare_scaled[Fare_scaled[:,0].argsort()][-10:]
print("Outer range (low) of the distribution:")
print(low_range)
print("\n Outer range (High) of the distribution:")
print(high_range)


# In[ ]:





# # Bivariate Analysis

# In[23]:


var = 'area'
data = pd.concat([df_train['price'],df_train[var]],axis= 1)
data.plot.scatter(x=var,y='price',ylim=(0));


# In[24]:


#delete the points
#df_train.sort_values(by ='area',ascending = False)[:1]
#df_train = df_train.drop(df_train[''])


# In[25]:


#scatter plot totalbsmtsf/price
var = 'area'
data = pd.concat([df_train['price'],df_train[var]], axis =1)
data.plot.scatter(x=var, y='price',ylim=(0));


# In[26]:


plt.figure(figsize=(20,8))

plt.subplot(1,2,1)
plt.title('House Price Distribution Plot')
sns.distplot(df_train.price)

plt.subplot(1,2,2)
sns.boxplot(df_train.price)
plt.title('House Pricing Spread')

plt.show()


# In[27]:


categorical_list = [x for x in df_train.columns if df_train[x].dtype == 'object']
for x in categorical_list: print(x)


# In[28]:


plt.figure(figsize=(15, 6))

plt.subplot(1,3,1)
plt1 = df_train['mainroad'].value_counts().plot(kind='bar')
plt.title('mainroad Histogram')
plt1.set(xlabel = 'mainroad', ylabel='Frequency of mainroad')

plt.subplot(1,3,2)
plt1 = df_train['guestroom'].value_counts().plot(kind='bar')
plt.title('Guestroom Histogram')
plt1.set(xlabel = 'Guestroom', ylabel='Frequency of Guestroom')

plt.subplot(1,3,3)
plt1 = df_train['basement'].value_counts().plot(kind='bar')
plt.title('basement Histogram')
plt1.set(xlabel = 'Basement', ylabel='Frequency of Basement')

plt.show()


# In[29]:


df_train


# In[30]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("HotwaterHeating vs price")
sns.boxplot(x=df_train.hotwaterheating, y=df_train.price, palette=('cubehelix'))

plt.subplot(1,2,2)
plt.title("Basement vs price")
sns.boxplot(x=df_train.basement, y=df_train.price, palette=('PuBuGn'))

plt.show()


# In[31]:


plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title('Mainroad vs Price')
sns.boxplot(x=df_train.mainroad, y=df_train.price, palette=("cubehelix"))

plt.subplot(1,2,2)
plt.title('guestroom vs Price')
sns.boxplot(x=df_train.guestroom, y=df_train.price, palette=("PuBuGn"))

plt.show()


# In[32]:


plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.title('airconditioning vs Price')
sns.boxplot(x=df_train.airconditioning, y=df_train.price, palette=("cubehelix"))

plt.subplot(1,3,2)
plt.title('prefarea vs Price')
sns.boxplot(x=df_train.prefarea, y=df_train.price, palette=("PuBuGn"))

plt.subplot(1,3,3)
plt.title('furnishingstatus vs Price')
sns.boxplot(x=df_train.furnishingstatus, y=df_train.price, palette=("PuBuGn"))
plt.xticks(rotation=45)

plt.show()


# In[ ]:




