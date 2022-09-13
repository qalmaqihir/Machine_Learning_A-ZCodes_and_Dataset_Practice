
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
dataset=pd.read_csv("../../Datasets/Data.csv")
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

# Checking
print(X)
print(y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

print(X)


# Encoding categorical data
# Encoding the independent variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#Encoding the dependent variables
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y)
print(y)



#Splitting the dataset into Training and Testing Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=0)

print(f"X_Train\n{X_train}")
print(f"X_Test\n{X_test}")
print(f"y_Train\n{y_train}")
print(f"y_Test\n{y_test}")

print("Shapes\n\n")

print(f"X_Train shape\n{X_train.shape}")
print(f"X_Test shape\n{X_test.shape}")
print(f"y_Train shape\n{y_train.shape}")
print(f"y_Test shape\n{y_test.shape}")

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.transform(X_test[:,3:])

print(X_train)
print(X_test)