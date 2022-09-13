
# XG_boost


# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_path="../Datasets/Social_Network_Ads.csv"
dataset=pd.read_csv(dataset_path)
print(dataset.head())
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

# Spliting the dataset into the training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,  y_test = train_test_split(X,y, test_size=0.25)

print(f"Lenght of x_test {len(X_test)}\nlenght of y_test {len(y_test)}")


# feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# Training the xg_boost model on the training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
# Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)

print(accuracy_score(y_pred,y_test))

# Applying k fold cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train,y_train,cv=5)
print(scores)
print(f"Mean Accuracy : {scores.mean()*10000//100}%")
print(f"Standard Deviation: {scores.std()*10000//100}")
#
# # Visualizing the Training set reuslt
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression(Training set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Visualising the test result
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()