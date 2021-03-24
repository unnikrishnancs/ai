#!/usr/bin/env python

"""
2. Print the Accuracy Score and Confusion matrix for KNN Classifier using iris data.
(Split iris dataset to train and test sets.)
"""

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection as ms
from sklearn.metrics import confusion_matrix,accuracy_score

iris_df=load_iris()

X=iris_df.data
y=iris_df.target

#incase of 'import'
X_train,X_test,y_train,y_test=ms.train_test_split(X,y,test_size=0.2) 

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)

#Apply knn on X_test and predict value of y_test
predictd_vals_of_y=knn.predict(X_test) 

#Predicted value of y_test
print("")
print("Predicted values of y:")
print(predictd_vals_of_y) 

#Actual value of y_test
print("")
print("Actual values of y:")
print(y_test) 

#Compare actual values (y_test) and predicted values(predictd_vals_of_y)
print("")
print("Confusion_matrix :To compare values of Actual(y_test) and Predicted values")
print(confusion_matrix(y_test,predictd_vals_of_y)) 

#Compare the accuracy score of actual values(y_test) and predicted values(prediction)
print("")
print("Accuracy_score")
print(accuracy_score(y_test,predictd_vals_of_y)) 
print("")
