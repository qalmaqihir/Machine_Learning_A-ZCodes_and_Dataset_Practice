
# Grid search

# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_path="/home/qalmaqihir/BreakGojalti/current/codes_for_books/Machine Learning A-Z (Codes and Datasets)/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv"
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

# Training the Kernl svm model on the training set
from sklearn.svm import SVC
classifier=SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,y_train)

# Predicting the test results
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

# Applying grid search to find the best model and best parameters
from sklearn.model_selection import GridSearchCV
paramters=[{'C':[0.25,0.5,0.75,1],'kernel':['rbf'], 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]},
           {'C':[0.25,0.5,0.75,1],'kernel':['linear']}]

gs=GridSearchCV(estimator=classifier,param_grid=paramters,scoring='accuracy',cv=10,n_jobs=-1)
gs.fit(X_train,y_train)
best_accuracy=gs.best_score_
best_parameters = gs.best_params_
print(f"Best Accuracy: {best_accuracy}")
print(f"Best Parameters: {best_parameters}")
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
plt.title('Kernel SVC (Training set)')
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
plt.title('Kernel SVC (Test set)')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()