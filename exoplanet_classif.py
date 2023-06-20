import numpy as np
import pandas as pd

#classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier

#metrics and stuff
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv("exoplanet_data.csv")

def fill_bool(row):
    if row == "Yes":
        return 1
    return 0
data["star_magnetic_field"] = data["star_magnetic_field"].apply(fill_bool)

for column in data: 
    data[column].fillna(-1, inplace = True)

x = data.drop(columns=["# name", "planet_status", "discovered", "updated", "star_sp_type", "star_detected_disc"], axis=1)
y = data["planet_status"]
train_data, test_data, train_labels, test_labels = train_test_split(x, y , test_size=0.2)

# Create an instance of the Gaussian Naive Bayes classifier
classifier = GaussianNB()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix1: ", confusion_matrix(test_labels, predictions))
print("Accuracy1: ", accuracy_score(test_labels, predictions)*100)
 

"""
# Create an instance of the GaussianProcessClassifier
classifier = GaussianProcessClassifier()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix2: ", confusion_matrix(test_labels, predictions))
print("Accuracy2: ", accuracy_score(test_labels, predictions)*100)
"""
# Create an instance of the RandomForestClassifier
classifier = RandomForestClassifier()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix3: ", confusion_matrix(test_labels, predictions))
print("Accuracy3: ", accuracy_score(test_labels, predictions)*100)



