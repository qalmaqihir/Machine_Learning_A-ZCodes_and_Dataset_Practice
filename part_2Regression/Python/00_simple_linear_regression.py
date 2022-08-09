# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X= dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


# Splitting the dataset into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

# Training the Simple Linear Regression model on the training set
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set result
y_pred  = regressor.predict(X_test)



# Visualising funtion
def visualize_results(x,y,title,x_label,y_label):
    plt.scatter(x,y,color='red')
    plt.plot(x,regressor.predict(x),color="blue")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# Visualising the training set result
visualize_results(X_train,y_train,"Salary Vs Experience - Training Set","Years of Experience","Salary")

# Visualising the testing set result
visualize_results(X_test,y_test,"Salary Vs Experience - Testing Set","Years of Experience","Salary")

