# Polynomial Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
path="/home/qalmaqihir/BreakGojalti/current/codes_for_books/Machine Learning A-Z (Codes and Datasets)/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv"

dataset = pd.read_csv(path)
X= dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values

print(X)
print(y)
# # Splitting the dataset into training and testing set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

# Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
model=lin_reg.fit(X,y)

print(model)
# Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

# Visualising the Linear Regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()

# Visualsing the Polynomial Regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title("Truth or Bluff (Polynomail Regression)")
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()



# Visualising the Polynomail Regression results (for higher resolution and smoother curve)
X_grid=np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title("Truth or Bluff (Polynomail Regression)")
plt.xlabel('Position Level')
plt.ylabel("Salary")
plt.show()


# Predicting a new result with linear Regression
pred_l=lin_reg.predict([[6.5]])

# Predicting a new result with Polynomail Regression
pred_ml=lin_reg2.predict(poly_reg.fit_transform([[6.5]]))

print(f"Simple Linear Regression prediction result {pred_l}\nMultiple Linear Regression Prediction result {pred_ml}")