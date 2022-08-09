
"""
This is the data preprocessing Template
The basic steps are:
1. Importing the libraries
2. Importing the dataset (dependent and independent variables, x,y)
3. Splitting the dataset into the training and testing set
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv("/home/qalmaqihir/BreakGojalti/current/codes_for_books/Machine Learning A-Z (Codes and Datasets)/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Python/Data.csv")
X=dataset.iloc[:,:-1].values # We can use loc with colunm names specified? not working
# X= dataset.loc(["Country","Age","Salary"])
y=dataset.iloc[:,-1].values
print(dataset)
print(X)
print(y)

#Splitting the dataset into Training and Testing Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,random_state=0)

print("Printing the type and shape of X train/test")
print(X_train.shape)
print(type(X_train))
print("Printing the type and shape of y train/test")
print(y_train.shape)
print(type(y_train))
