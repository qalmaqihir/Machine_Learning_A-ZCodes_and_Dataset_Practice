
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
print("\nX_train\n")
print(X_train)
print("\ny_train\n")
print(y_train)

print(f"Lenght of x_test {len(X_test)}\nlenght of y_test {len(y_test)}")


# feature scaling
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train = sc.fit_transform(X_train)
X_test=sc.transform(X_test)

print(f"After feautre scaling the X_train\n{X_train}\n")
print(f"After feautre scaling the X_test\n{X_test}\n")

# Training the Kernl svm model on the training set
from sklearn.svm import SVC
classifier=SVC(kernel='rbf', random_state=0)
classifier.fit(X_train,y_train)

# Predicting a new result
print(f" A prediction {classifier.predict(sc.transform([[30,87000]]))}")

# Predicting the test results
y_pred = classifier.predict(X_test)
print("Side by side")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1))))

# Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)

print(accuracy_score(y_pred,y_test))


# # Visualising the test result
# from matplotlib.colors import ListedColormap
# X_set, y_set = sc.inverse_transform(X_test), y_test
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = 0.25),
#                      np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = 0.25))
# plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('red', 'green')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
# plt.title('Logistic Regression (Test set)')
# plt.xlabel('Age')
# plt.ylabel('Estimated Salary')
# plt.legend()
# plt.show()