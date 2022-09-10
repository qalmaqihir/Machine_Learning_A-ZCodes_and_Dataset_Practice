
# ANN
# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Part - I Data Preprocessing
# Importing Dataset
dataset_path="/home/qalmaqihir/BreakGojalti/current/codes_for_books/Machine Learning A-Z (Codes and Datasets)/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv"
dataset = pd.read_csv(dataset_path)
print(dataset.head())
x = dataset.iloc[:,3:-1].values
y= dataset.iloc[:,-1].values
print(x)
print(y)

# Encoding categorical data
# Encoding the female/male features
# Label Encoding of the 'Gender' column
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
print(x)

# One Hot Encoding the  'Geography' column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)


# Splitting the dataset into Training and Testing Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
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
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

print(X_train)
print(X_test)


# Part - II Building the ANN
# initialize the ANN, as sequence of layers
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=16, activation='relu'))

# adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# Part -III Training the ANN
# Compile the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test,y_test))


# Part -IV Making the predictions and evaluating the model
data_point = [['France', 600, 'Male',40,3,60000,2,1,1,50000]]

# Ecoding the gender
data_point[0,2]=le.fit_transform(data_point[0,2])
# encoding the geography
ct2 = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
data_point = np.array(ct2.fit_transform(data_point))

# Feature scaling on the data point
data_point=sc.transform(data_point)
result = ann.predict(data_point)
print(result>0.5)

# Predicting the testing result
y_pred = ann.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),  y_test.reshape(len(y_test),1)),1))

# Make the confusion matrix
