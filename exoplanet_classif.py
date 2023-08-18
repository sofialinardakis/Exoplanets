import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

#metrics and stuff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv("exoplanet_data.csv")

def fill_bool(row):
    if row == "Yes":
        return 1
    return 0
data["star_magnetic_field"] = data["star_magnetic_field"].apply(fill_bool)

def fill_conf(row):
    if row == "Confirmed":
        return 1
    return 0
data["planet_status"] = data["planet_status"].apply(fill_conf)


for column in data: 
    data[column].fillna(-1, inplace = True)
    data[column].replace("", -1, inplace=True)

#data.to_csv('updated_exoplanet_data.csv', index=False)



x = data.drop(columns=["# name", "planet_status", "discovered", "updated", "star_sp_type", "star_detected_disc"], axis=1)
y = data["planet_status"]
train_data, test_data, train_labels, test_labels = train_test_split(x, y , test_size=0.2)

"""
plt.hist(data["radius"], bins=20, edgecolor="black")
plt.xlabel("Radius")
plt.ylabel("Freq")
plt.title("Distribution of exoplanet radii")
plt.show()
import seaborn as sns


correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation heatmap")
plt.show()

sns.boxplot(x='star_age', y='planet_status', data=data)
plt.xlabel('age')
plt.ylabel('stat')
plt.title('Exoplanet Radii by age ')
plt.show()
"""
"""
import plotly.express as px

# Example: Scatter plot with hover information
fig = px.scatter(data, x='orbital_period', y='mass', color='radius', hover_data=['# name'])
fig.update_layout(title='Exoplanet Mass vs. Orbital Period', xaxis_title='Orbital Period', yaxis_title='Mass')
fig.show()
"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_data)
X_test_scaled = scaler.transform(test_data)

# Create an instance of the Gaussian Naive Bayes classifier
classifier = GaussianNB()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix1: \n", confusion_matrix(test_labels, predictions))
print("Accuracy1: ", accuracy_score(test_labels, predictions)*100)


 
 
# Create an instance of the GaussianProcessClassifier
classifier = GaussianProcessClassifier()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix2: \n", confusion_matrix(test_labels, predictions))
print("Accuracy2: ", accuracy_score(test_labels, predictions)*100)



# Create an instance of the RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state = 42)
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix3: \n", confusion_matrix(test_labels, predictions))
print("Accuracy3: ", accuracy_score(test_labels, predictions)*100)


# Create an instance of the LogisticRegression
classifier = LogisticRegression(max_iter=1000)
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix4: \n", confusion_matrix(test_labels, predictions))
print("Accuracy4: ", accuracy_score(test_labels, predictions)*100)


# Create an instance of the SVC
classifier = SVC()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix5: \n", confusion_matrix(test_labels, predictions))
print("Accuracy5: ", accuracy_score(test_labels, predictions)*100)


# Create an instance of the KNeighborsClassifier
classifier = KNeighborsClassifier()
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix6: \n", confusion_matrix(test_labels, predictions))
print("Accuracy6: ", accuracy_score(test_labels, predictions)*100)


# Create an instance of the MLPClassifier
classifier = MLPClassifier(max_iter=1000)
# Fit the classifier to the training data
classifier.fit(train_data, train_labels)
# Make predictions on the test data
predictions = classifier.predict(test_data)
# Evaluate the classifier's performance
print("Matrix7: \n", confusion_matrix(test_labels, predictions))
print("Accuracy7: ", accuracy_score(test_labels, predictions)*100)


"""
classifier = LinearRegression()
classifier.fit(train_data, train_labels)
predictions = classifier.predict(test_data)
plt.scatter(data["radius"], data["planet_status"])
plt.title("Correlation")
plt.xlabel("rad")
plt.ylabel("stat")
plt.show()
#no clear correlations observed"""

"""
dat = pd.read_csv("updated_exoplanet_data.csv")
#new dataset with only confirmed
filtered = dat[dat["planet_status"] == 1]
# Print the filtered exoplanet information
#print(filtered)
filtered.to_csv('confirmed.csv', index=False)

"""
"""
import seaborn
exo_data = pd.read_csv("confirmed.csv")

subset_data = exo_data.sample(frac=0.03, random_state=42)
#subset_data.replace(-1, pd.NA, inplace=True)
seaborn.pairplot(
    subset_data,
    x_vars=["mass", "radius", "orbital_period", "star_distance", "star_mass", "star_age"],
    y_vars=["mass", "radius", "orbital_period", "star_distance", "star_mass", "star_age"],
    hue="planet_status",
    kind="scatter"
)
plt.show()
"""





"""
subset_data.replace(-1, pd.NA, inplace=True)

subset_data = subset_data.dropna()
seaborn.regplot(x="mag_k", y="mag_v",
             data=subset_data)

plt.show()"""




"""
import scipy.stats as stats
correlation_matrix = subset_data.corr()
print(correlation_matrix)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()"""