import numpy as np
import matplotlib.pyplot as ply
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

def main():
    dataset = pd.read_csv('dataCSV.csv')
    X = dataset.iloc[:, :-1].values # all columns except last one, matrix of features x that predict y
    y = dataset.iloc[:, -1].values # only last column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    le = preprocessing.LabelEncoder()
    le.fit(dataset.iloc[:, -1].values)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)

    model = pickle.load(open('random_forest_model.sav', 'rb'))

    data = open("./TrainingData/TrainingDataWords/Down10.txt", "r")

    lines = data.readlines()

    res = []

    for line in lines:
        line = line.strip("\n")
        line = line.replace(" ", "")
        str1 = line.replace(']','').replace('[','')
        l = str1.replace('"','').split(",")
        l = [int(numeric_string) for numeric_string in l]

        res.append(le.inverse_transform(model.predict(sc.transform([l]))))

    print(most_frequent(res))

def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 

main()