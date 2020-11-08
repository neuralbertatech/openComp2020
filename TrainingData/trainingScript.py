from pyOpenBCI import OpenBCICyton
import time
import random
import sys
import threading
# import Queue
import os
import numpy as np
import matplotlib.pyplot as plt

DataCounter = 501
<<<<<<< HEAD
StartingNumber = 1000#680
SamplesToRecord = 0#20
=======
StartingNumber = 180
SamplesToRecord = 20
>>>>>>> 1c80e3685df64a0f763022f676447e3a938b6ba8
EndingNumber = -1 # assign to -1 to be automatically determined by SamplesToRecord
TestSubject = "Cameron"
TestFeature = "Limbs"
FileNumber = [StartingNumber, StartingNumber, StartingNumber, StartingNumber, StartingNumber, StartingNumber, StartingNumber, StartingNumber]
FileTypes = ["Up", "Down", "Forward", 'Back', "Left", "Right", "Stay", "Jaw"]
CurrFileType = 0
<<<<<<< HEAD
LastFileType = 0
GeneratePlots = True
MyFile = open(TestSubject + TestFeature + "/" + FileTypes[CurrFileType] + str(FileNumber[CurrFileType]) + ".txt", 'w')
=======
MyFile = open(TestSubject + TestFeature + "/" + FileTypes[CurrFileType] + str(FileNumber[CurrFileType]) + ".txt", 'w')

if(EndingNumber == -1):
    EndingNumber = StartingNumber + SamplesToRecord
>>>>>>> 1c80e3685df64a0f763022f676447e3a938b6ba8

if(EndingNumber == -1):
    EndingNumber = StartingNumber + SamplesToRecord

# def add_input(input_queue):
#     while True:
#         input_queue.put(sys.stdin.read(1))

<<<<<<< HEAD


def plotSample(filePath):
    global FileTypes, CurrFileType
    channels = [[] for i in range(16)]

    with open(filePath, "r") as file:

        # Format data
        for line in file:
            #Remove "[]" and convert to an array
            line = line[1:-2].split(", ")
            line.pop()

            # Convert strings to ints and convert to channel-wise data
            for i in range(len(line)):
                channels[i].append(int(line[i]))

    # Plot the data
    plt.clf()
    x = np.linspace(0,len(channels[0])-1, len(channels[0]))

    plotColors = ["#ff0000", "#ff0038", "#ff0060", "#ff0085", "#ff26a9", "#ff4aca", "#ee65e7", "#d47bff", "#b68fff", "#92a0ff", "#6aaeff", "#38bbff", "#00c5ff", "#00ceff", "#00d6ff", "#00dcff"]
=======
    if(DataCounter < 250): #250 samples = 1 second (250Hz)
        dataPoint = var.channels_data
        dataPoint.append(str(FileTypes[CurrFileType]))

        MyFile.write(str(dataPoint) + "\n")
        MyFile.flush()
        DataCounter += 1

    else:
        if(DataCounter > 500): # First run, ignore it
            DataCounter = 0
            CurrFileType = random.randint(0,7)

        else:
            FileNumber[CurrFileType] += 1

            CurrFileType = random.randint(0,7)

            print("Current Samples: ")
            print(FileNumber)

            if(all(i > EndingNumber for i in FileNumber)):
                exit()

            while(FileNumber[CurrFileType] > EndingNumber):
                CurrFileType = random.randint(0,7)

            DataCounter = 0
            MyFile = open(TestSubject + TestFeature + "/" + FileTypes[CurrFileType] + str(FileNumber[CurrFileType]) + ".txt", 'w')

            print("### " + FileTypes[CurrFileType] + " ###")
            time.sleep(1)
            print("Recording.")
>>>>>>> 1c80e3685df64a0f763022f676447e3a938b6ba8

    for channel in range(len(channels)):
        plt.plot(x, channels[channel], plotColors[channel])
    plt.xlabel("Sample")
    plt.ylabel("μV")

    plt.savefig("Plots/" + str(FileTypes[CurrFileType]) + ".png")



def print_raw(var):
    global DataCounter, FileNumber, FileTypes, CurrFileType, LastFileType, MyFile

    # print(var.channels_data)

    if(DataCounter < 250): #250 samples = 1 second (250Hz)
        dataPoint = var.channels_data
        dataPoint.append(str(FileTypes[CurrFileType]))

        MyFile.write(str(dataPoint) + "\n")
        MyFile.flush()
        DataCounter += 1

    else:
        if(DataCounter > 500): # First run, ignore it
            DataCounter = 0
            CurrFileType = random.randint(0,7)

        else:
            MyFile.close()
            if(GeneratePlots):
                if(TestSubject + TestFeature + "/" + FileTypes[CurrFileType] + str(FileNumber[CurrFileType]) + ".txt" != "CameronLimbs/Stay1000.txt"):
                    plotSample(TestSubject + TestFeature + "/" + FileTypes[CurrFileType] + str(FileNumber[CurrFileType]) + ".txt")

            FileNumber[CurrFileType] += 1
            LastFileType = CurrFileType
            CurrFileType = random.randint(0,7)

            print("Current Samples: ")
            print(FileNumber)

            if(all(i > EndingNumber for i in FileNumber)):
                exit()

            while(FileNumber[CurrFileType] > EndingNumber):
                CurrFileType = random.randint(0,7)

            DataCounter = 0
            MyFile = open(TestSubject + TestFeature + "/" + FileTypes[CurrFileType] + str(FileNumber[CurrFileType]) + ".txt", 'w')

            print("### " + FileTypes[CurrFileType] + " ###")
            time.sleep(1)
            print("Recording.")


    # # User inputted that the sample failed, remove sample.
    # if not input_queue.empty():
    #     print("Deleting last recording of " + FileTypes[LastFileType] + " at " + TestSubject + TestFeature + "/" + FileTypes[LastFileType] + str(FileNumber[LastFileType]) + ".txt")
    #
    #     if os.path.exists(TestSubject + TestFeature + "/" + FileTypes[LastFileType] + str(FileNumber[LastFileType] - 1) + ".txt"):
    #       os.remove(TestSubject + TestFeature + "/" + FileTypes[LastFileType] + str(FileNumber[LastFileType] - 1) + ".txt")
    #       FileNumber[LastFileType] -= 1
    #       print("Successfully deleted.\n")
    #     else:
    #       print("Some error occurred and the sample was not deleted.")
    #
    #     input_queue.get() #empty queue

# input_queue = Queue.Queue()
#
# input_thread = threading.Thread(target=add_input, args=(input_queue,))
# input_thread.daemon = True
# input_thread.start()

board = OpenBCICyton(port='/dev/tty.usbserial-DM01N7JO', daisy=True)

board.start_stream(print_raw)

# Word
## Arrows (Seeing on Screen)
## Colour
## Letter
## Moving Body Part
# Drone Movement
