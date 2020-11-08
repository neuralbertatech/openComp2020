from pyOpenBCI import OpenBCICyton
import time
import numpy as np
import matplotlib.pyplot as ply
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

Data = []
DataCounter = 0

Variables = pickle.load(open('random_forest_model_variables.sav', 'rb'))
le = Variables['le']
sc = Variables['sc']

Model = pickle.load(open('random_forest_model.sav', 'rb'))

def main():
    # variables = pickle.load(open('random_forest_model_variables.sav', 'rb'))
    # le = variables['le']
    # sc = variables['sc']
    #
    # model = pickle.load(open('random_forest_model.sav', 'rb'))


    # Pre Recorded Data
    # data = open("./TrainingData/TrainingDataWords/Back13.txt", "r")
    # lines = data.readlines()

    # Feed in live OpenBCI data here
    board = OpenBCICyton(port='/dev/tty.usbserial-DM01N7JO', daisy=True)

    print('Reading in')
    print('3...')
    time.sleep(1)
    print('2...')
    time.sleep(1)
    print('1...')
    time.sleep(1)

    board.start_stream(collect_sample)


    # res = []
    #
    # for line in lines:
    #     line = line.strip("\n")
    #     line = line.replace(" ", "")
    #     str1 = line.replace(']','').replace('[','')
    #     l = str1.replace('"','').split(",")
    #     l = [int(numeric_string) for numeric_string in l]
    #
    #     res.append(le.inverse_transform(model.predict(sc.transform([l]))))
    #
    # print(most_frequent(res))

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num



def collect_sample(var):
    global DataCounter

    # print(var.channels_data)

    if(DataCounter < 250): #500 samples = 2 seconds (250Hz)
        Data.append(var.channels_data)
        DataCounter += 1

    else:
        classify()
        DataCounter = 0
        Data.clear()

def classify():
    # Data collected! Make Prediction

    res = []

    for line in Data:
        # line = line.strip("\n")
        # line = line.replace(" ", "")
        # str1 = line.replace(']','').replace('[','')
        # l = str1.replace('"','').split(",")
        # l = [int(numeric_string) for numeric_string in l]

        res.append(le.inverse_transform(Model.predict(sc.transform([line]))))
        print(le.inverse_transform(Model.predict(sc.transform([line]))))

    print("############")
    print(most_frequent(res))
    print("############")

main()
