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
    variables = pickle.load(open('random_forest_model_variables.sav', 'rb'))
    le = variables['le']
    sc = variables['sc']

    model = pickle.load(open('random_forest_model.sav', 'rb'))

    data = open("./TrainingData/TrainingDataWords/Back13.txt", "r")

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