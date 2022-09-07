#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Apriori

# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import apyori


# In[3]:


# Data Preprocessing
dataset_path="/home/qalmaqihir/BreakGojalti/current/codes_for_books/Machine Learning A-Z (Codes and Datasets)/Part 5 - Association Rule Learning/Section 28 - Apriori/Python/Market_Basket_Optimisation.csv"
dataset = pd.read_csv(dataset_path,header=None)
print(dataset.head())

tranaction=[]
for i in range(0,7501):
    tranaction.append([dataset.values[i,j] for j in range(0,20)])

tranaction
# Training the Apriori Model on the dataset


# In[4]:


# Training the Apriori Model on the dataset
from apyori import apriori


# In[12]:


rules=apriori(transactions=tranaction, min_support=0.003,
             min_confidence=0.2, min_lift=3,min_lenght=2, max_lenght=2)


# In[13]:


# # Visulize the results
# results=list(rules)
# results


# In[14]:


# Apriori

# Run the following command in the terminal to install the apyori package: pip install apyori

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[17]:


# Data Preprocessing
dataset = pd.read_csv(dataset_path, header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])


# In[18]:


transactions


# In[19]:


# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)


# In[20]:


rules


# In[21]:


# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted
resultsinDataFrame


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




