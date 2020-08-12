from pyOpenBCI import OpenBCICyton
import time
import random

DataCounter = 501
FileNumber = [-1, 0, 0, 0, 0, 0, 0] # first has to be -1
FileTypes = ["Up", "Down", "Forward", 'Back', "Left", "Right", "Stay"]
CurrFileType = 0
MyFile = open("TrainingDataDroneMovement/" + FileTypes[CurrFileType] + str(FileNumber[CurrFileType]) + ".txt", 'w')

def print_raw(var):
    global DataCounter, FileNumber, FileTypes, CurrFileType, MyFile

    # print(var.channels_data)

    if(DataCounter < 500):
        MyFile.write(str(var.channels_data) + "\n")
        MyFile.flush()
        DataCounter += 1

    else:
        FileNumber[CurrFileType] += 1

        CurrFileType = random.randint(0,6)

        print("Current Samples: ")
        print(FileNumber)

        if(all(i > 15 for i in FileNumber)):
            crashprogram

        while(FileNumber[CurrFileType] > 15):
            CurrFileType = random.randint(0,6)

        DataCounter = 0
        MyFile = open("TrainingDataDroneMovement/" + FileTypes[CurrFileType] + str(FileNumber[CurrFileType]) + ".txt", 'w')

        print("Think " + FileTypes[CurrFileType])
        print("3")
        time.sleep(1)
        print("2")
        time.sleep(1)
        print("1")
        time.sleep(1)



board = OpenBCICyton(port='/dev/tty.usbserial-DM01N7JO', daisy=True)

board.start_stream(print_raw)

# Word
## Arrows (Seeing on Screen)
## Colour
## Letter
## Moving Body Part
# Drone Movement
