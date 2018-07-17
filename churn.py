import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from datetime import datetime

print("Importing data")
churn_df = pd.read_csv('data/churn.csv')
print(churn_df.head(6))

print("Preparing features and labels")
# Isolate target data
y = churn_df['churn_or_not']

# We don't need these columns
to_drop = ['uid','churn_or_not']
churn_features = churn_df.drop(to_drop, axis=1)

def date_transform(register_date):
    dt = datetime.strptime(register_date, "%m/%d/%y")
    today = datetime(2018, 7, 17)
    delta = today - dt
    return str(delta.days)
churn_features['register_date'] = churn_features['register_date'].apply(date_transform)

X = churn_features.values.astype(np.float)

# This is important
scaler = StandardScaler()
X = scaler.fit_transform(X)

print("Feature space holds %d observations and %d features" % X.shape)
print("Unique target labels: " + str(np.unique(y)))

example_num = len(X)
train_num = int(example_num * 2 / 3)
train_index = range(0, train_num)
test_index = range(train_num, example_num)
print("Splitting train and test data set: %d, %d" % (len(train_index), len(test_index)))

X_train, y_train = X[train_index], y[train_index]
X_test, y_test = X[test_index], y[test_index]


def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1. and 0.
    return np.mean(y_true == y_pred)


print("Support vector machines")
clf = SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: %.3f" % accuracy(y_test, y_pred))

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print "Precision: %.3f, Recall: %.3f" % (1.0 * matrix[1][1] / (matrix[0][1] + matrix[1][1]), 1.0 * matrix[1][1] / (matrix[1][0] + matrix[1][1]))


print("K-nearest-neighbors")
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: %.3f" % accuracy(y_test, y_pred))

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print("Precision: %.3f, Recall: %.3f" % (1.0 * matrix[1][1] / (matrix[0][1] + matrix[1][1]), 1.0 * matrix[1][1] / (matrix[1][0] + matrix[1][1])))


print("Random forest")
clf = RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: %.3f" % accuracy(y_test, y_pred))

matrix = confusion_matrix(y_test, y_pred)
print(matrix)
print("Precision: %.3f, Recall: %.3f" % (1.0 * matrix[1][1] / (matrix[0][1] + matrix[1][1]), 1.0 * matrix[1][1] / (matrix[1][0] + matrix[1][1])))

# Write test data to file
churn_test = churn_features.ix[test_index]
churn_test['churn_or_not'] = y_test
churn_test['Predicted'] = y_pred
churn_test.to_csv("./data/churn_test.csv", index=False)
