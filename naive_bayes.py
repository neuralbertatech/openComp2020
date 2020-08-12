import numpy as np
import matplotlib.pyplot as ply
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

def main():
    dataset = pd.read_csv('dataCSV.csv')
    X = dataset.iloc[:, :-1].values # all columns except last one, matrix of features x that predict y
    y = dataset.iloc[:, -1].values # only last column

    le = preprocessing.LabelEncoder()
    le.fit(dataset.iloc[:, -1].values)

    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(accuracy_score(y_test, y_pred))



main()