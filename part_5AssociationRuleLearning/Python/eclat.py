# Elcat Model

# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import apyori

# Data Preprocessing
dataset_path="../../Datasets/Market_Basket_Optimisation.csv"
dataset = pd.read_csv(dataset_path,header=None)
print(dataset.head())

transaction=[]
for i in range(0,7501):
    transaction.append([dataset.values[i,j] for j in range(0,20)])

print(transaction)
# Training the Apriori Model on the dataset
rules=apriori(transactions=transaction, min_support=0.003,
             min_confidence=0.2, min_lift=3,min_lenght=2, max_lenght=2)
