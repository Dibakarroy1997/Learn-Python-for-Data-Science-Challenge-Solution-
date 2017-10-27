#importing libraries
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import seaborn as sns
import pandas as pd

#Declaring classifer object
clf_tree = tree.DecisionTreeClassifier()
clf_svm = SVC()
clf_KNC = KNeighborsClassifier()
clf_GPC = GaussianProcessClassifier()

#Here we are taking X as fetures and Y as labels
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ["male", "male", "female", "female", "male", "male", "female", "female", "female", "male", "male"]

#Training the classifier using the data
clf_tree = clf_tree.fit(X, Y)
clf_svm = clf_svm.fit(X, Y)
clf_KNC = clf_KNC.fit(X, Y)
clf_GPC = clf_GPC.fit(X, Y)

#Predict the data using classifier
prediction_tree = clf_tree.predict(X)
prediction_svm = clf_svm.predict(X)
prediction_KNC = clf_KNC.predict(X)
prediction_GPC = clf_GPC.predict(X)

#Check how accurate the data is
acu_tree = accuracy_score(Y, prediction_tree) * 100
acu_svm = accuracy_score(Y, prediction_svm) * 100
acu_KNC = accuracy_score(Y, prediction_KNC) * 100
acu_GPC = accuracy_score(Y, prediction_GPC) * 100

#Display the accuracy
print("Accuracy of DecisionTreeClassifier: {}%".format(acu_tree))
print("Accuracy of Decision Support Vector Classification: {}%".format(acu_svm))
print("Accuracy of Decision KNeighborsClassifier: {}%".format(acu_KNC))
print("Accuracy of Decision GaussianProcessClassifier: {}%".format(acu_GPC))