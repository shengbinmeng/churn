from datetime import datetime
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline

print("Importing data")
churn_df = pd.read_csv('data/churn.csv')
print(churn_df)

print("Preparing features and labels")
churn_labels = churn_df['churn_or_not']
churn_features = churn_df.drop(['uid', 'churn_or_not'], axis=1)


# Transform date to days.
def date_transform(register_date):
    dt = datetime.strptime(register_date, "%m/%d/%y")
    today = datetime(2018, 6, 30)
    delta = today - dt
    return str(delta.days)


churn_features['register_date'] = churn_features['register_date'].apply(date_transform)

X = churn_features.values.astype(np.float)
y = churn_labels.values

print("We have %d examples, each with %d features" % X.shape)
print("Unique labels: " + str(np.unique(y)))

example_num = len(X)
train_num = int(example_num * 8 / 10)
train_index = list(range(0, train_num))
test_index = list(range(train_num, example_num))
print("Splitting train and test data set: %d, %d" % (len(train_index), len(test_index)))

X_train, y_train = X[train_index], y[train_index]
X_test, y_test = X[test_index], y[test_index]


def accuracy(y_true, y_pred):
    # NumPy interprets True and False as 1 and 0.
    return np.mean(y_true == y_pred)


def train(cls, features, labels, model_filename):
    pipe = make_pipeline(StandardScaler(), cls)
    pipe.fit(features, labels)
    # Store model.
    with open(model_filename, "wb") as model_file:
        pickle.dump(pipe, model_file)


def test(model_filename, features, labels):
    # Load back the trained model and use it for test.
    with open(model_filename, "rb") as model_file:
        trained_pipe = pickle.load(model_file)
    labels_pred = trained_pipe.predict(features)
    print("Accuracy: %.3f" % accuracy(labels, labels_pred))
    matrix = confusion_matrix(labels, labels_pred)
    print(matrix)
    print("Precision: %.3f, Recall: %.3f" % (1.0 * matrix[1][1] / (matrix[0][1] + matrix[1][1]), 1.0 * matrix[1][1] / (matrix[1][0] + matrix[1][1])))


print("Support vector machine")
train(SVC(), X_train, y_train, "SVM-model.pickle")
test("SVM-model.pickle", X_test, y_test)

print("Random forest")
train(RandomForestClassifier(), X_train, y_train, "RF-model.pickle")
test("RF-model.pickle", X_test, y_test)

print("K-nearest-neighbors")
train(KNeighborsClassifier(), X_train, y_train, "KNN-model.pickle")
test("KNN-model.pickle", X_test, y_test)


# Write SVM test result to file.
with open("SVM-model.pickle", "rb") as svm_model_file:
    svm_model = pickle.load(svm_model_file)
y_pred = svm_model.predict(X_test)
churn_test = churn_df.ix[test_index]
churn_test['Predicted'] = y_pred
churn_test.to_csv("churn_test.csv", index=False)
