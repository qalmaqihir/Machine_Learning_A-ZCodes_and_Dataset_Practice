# Apriori

# Import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import apyori

# Data Preprocessing
dataset_path="/home/qalmaqihir/BreakGojalti/current/codes_for_books/Machine Learning A-Z (Codes and Datasets)/Part 5 - Association Rule Learning/Section 28 - Apriori/Python/Market_Basket_Optimisation.csv"
dataset = pd.read_csv(dataset_path,header=None)
print(dataset.head())

tranaction=[]
for i in range(0,7501):
    tranaction.append([dataset.values[i,j] for j in range(0,20)])

print(tranaction)
# Training the Apriori Model on the dataset

