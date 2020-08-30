from pyOpenBCI import OpenBCICyton
import time
import random

DataCounter = 501
StartingNumber = 180
SamplesToRecord = 20
EndingNumber = -1 # assign to -1 to be automatically determined by SamplesToRecord
TestSubject = "Cameron"
TestFeature = "Limbs"
FileNumber = [StartingNumber, StartingNumber, StartingNumber, StartingNumber, StartingNumber, StartingNumber, StartingNumber, StartingNumber]
FileTypes = ["Up", "Down", "Forward", 'Back', "Left", "Right", "Stay", "Jaw"]
CurrFileType = 0
MyFile = open(TestSubject + TestFeature + "/" + FileTypes[CurrFileType] + str(FileNumber[CurrFileType]) + ".txt", 'w')

if(EndingNumber == -1):
    EndingNumber = StartingNumber + SamplesToRecord

def print_raw(var):
    global DataCounter, FileNumber, FileTypes, CurrFileType, MyFile

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



board = OpenBCICyton(port='/dev/tty.usbserial-DM01N7JO', daisy=True)

board.start_stream(print_raw)

# Word
## Arrows (Seeing on Screen)
## Colour
## Letter
## Moving Body Part
# Drone Movement
