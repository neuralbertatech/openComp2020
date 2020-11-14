from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from pyOpenBCI import OpenBCICyton
import matplotlib.pyplot as plt
import numpy as np
import threading
import tellopy
import random
import time
import sys
import os

"""
Here I will describe what this file does. (Maybe add how to run as well?)

If you have collected a baseline, but wish to rerun the program wihtout recollecting it, simply call:
python masterController.py nocol
"""


SAMPLE_LEN = 250                 # 250samples = 1s
SAMPLE_COUNT = 2
SAMPLE_RECORDING_TIME = 1        # Amount of samples to switch (time between recordings in units of 1/250s), (min=1)
SAVE_PLOTS = False
COLLECT_LIVE = True              # False will generate random data (reccomended to enable NO_FLY_MODE)
WAIT_TO_CONTINUE = False
TEST_TRAIN_SPLIT = 0.5
NO_FLY_MODE = True               # Connect drone, except instead of flying, just print output
SPEED = 20                       # value 0-100 that controls the drones speed

LDAModel = LinearDiscriminantAnalysis()

Dataset = []
ProcessedData = []
DatapointBeingCollected = [[] for i in range(16)]

DroneOnGround = True # Lets us know if we want to land or takeoff from a tkoff/lnd command
FirstRun = True # Ensures that we only start one drone thread
DroneConnected = False
CommandForDroneThread = ""
FlightDataOutput = ["", ""]

FileNumber = [-1, 0, 0, 0, 0, 0, 0, 0] # For some reason it just doesn't record the first Command, so start at -1
Commands = ["Up", "Down", "Forward", 'Back', "Left", "Right", "Stay", "Jaw"]
CurrCommand = 0
Count = SAMPLE_LEN+SAMPLE_RECORDING_TIME*2 + 1  # Used throughout code to keep a global count of iterations, set to large num to skip first iter of collecting

ProgramStep = 0 # 0 = collecting, 1 = training, 2 = flying

SampleSaveDirectory = "TrainingData/masterControllerSessions"  # Set to "" if you do not wish to record samples
# SampleSaveDirectory = ""
if(SampleSaveDirectory != ""):
    SampleFile = open(SampleSaveDirectory + "/" + Commands[CurrCommand] + str(FileNumber[CurrCommand]) + ".txt", 'w')

# NOTE: COLLECT_LIVE should be set to True if we choose to reuse data. COLLECT_LIVE = False will randomly control the drone.
SkipCollecting = False
if(len(sys.argv) > 1):
    if(str(sys.argv[1]) == "nocol"):
        if(SampleSaveDirectory != ""):
            SkipCollecting = True
        else:
            print("No sample directory defined, please update SampleSaveDirectory in masterController.py to utilize prerecorded samples, continuing.")
    else:
        print("Unknown argument provided, ignoring.")


def showLoadingBar(loadingCount, loadingTotal):
    """
    Takes in the current progress (loadingCount) and the total to display
    a loading bar. Returns an updated count if needed.
    """

    loadingBar = "#"*int(np.ceil(70*loadingCount/(loadingTotal)))
    sys.stdout.write('[%s] \r' % (loadingBar))
    sys.stdout.flush()

    return loadingCount + 1


def preprocess(training: bool, sampleNum = 0):
    """
    This func takes in a single data point and preprocesses it.
    This will work for both training and flying, so it needs to be fast and flexible
    - training tells us if we are referencing the dataset or classifying
    - sampleNum tells us which iter we are training
    """
    global Dataset, ProcessedData, Commands, DatapointBeingCollected

    print("#### Start Processing ####")   ###################################

    # [Delta (0-4), Theta (4-7.5), Alpha (7.5-12.5), Beta (12.5-30), Gamma (30-70)]
    bands = []

    for channelNum in range(16):

        # Appy a Savgol filter to smooth out imperfect data
        if(training):
            smoothedData = savgol_filter(Dataset[sampleNum][channelNum], 11, 2)
        else:
            smoothedData = savgol_filter(DatapointBeingCollected[channelNum], 11, 2)
        x = np.arange(0,125,1)


        if(SAVE_PLOTS and training): # We never want to plot while flying
            plt.clf()
            plt.plot(x, smoothedData, "r")
            plt.savefig("dataplt" + str(channelNum) + ".png")



        # FFT the data
        sp = np.fft.fft(smoothedData)
        freq = np.fft.fftfreq(x.shape[-1], 1/SAMPLE_LEN)

        freq = freq[1:int(np.ceil(SAMPLE_LEN/4))] # Only Care about positive
        sp = sp[1:int(np.ceil(SAMPLE_LEN/4))]
        sp = np.sqrt(sp.real**2 + sp.imag**2)



        if(SAVE_PLOTS and training): # We never want to plot while flying
            plt.clf()
            plt.plot(freq, sp, "r")
            plt.axvline(x=4, color="k")
            plt.axvline(x=7.5, color="k")
            plt.axvline(x=12.5, color="k")
            plt.axvline(x=30, color="k")
            plt.savefig("fft" + str(channelNum) + ".png")



        # Bin the results
        thisBand = [0,0,0,0,0]
        thisBandCount = [0,0,0,0,0]
        for point in range(len(freq)):
            if(freq[point] < 4):
                thisBand[0] += sp[point]
                thisBandCount[0] += 1
            elif(freq[point] < 7.5):
                thisBand[1] += sp[point]
                thisBandCount[1] += 1
            elif(freq[point] < 12.5):
                thisBand[2] += sp[point]
                thisBandCount[2] += 1
            elif(freq[point] < 30):
                thisBand[3] += sp[point]
                thisBandCount[3] += 1
            elif(freq[point] < 55): # To cut out powerline
                thisBand[4] += sp[point]
                thisBandCount[4] += 1

        # Append the average of all points in the bins
        bands.append(list(np.array(thisBand)/np.array(thisBandCount)))

    print("#### End Processing ####")    ###################################

    # Now, cast the set of bins of each electrode into a single set of bins (average)
    avgBands = []
    for bandNum in range(5):
        avgBands.append(round(np.average(np.array(bands)[:,bandNum]),2))

    if(training):
        # Add the label back
        avgBands.append(Dataset[sampleNum][-1])

        ProcessedData.append(avgBands)
    else:
        avgBands.append("")
        return avgBands


def checkForRailedChannels(data):
    global ProgramStep
    # print status of all channels (Railed, Good)

    allChannelsNotRailed = True # Encode this
    if(allChannelsNotRailed):
        print("No railed channels, begin collecting data")
        print("\n2: Collect Data")
        if(WAIT_TO_CONTINUE): input("Press return to continue...")
        ProgramStep += 1


def trainModel():
    global ProgramStep, ProcessedData, LDAModel, Commands

    ## In an ideal world, we will use cross validation
    # but since I dunno how to implement that rn and time is of the essence
    # we're gonna stick with naive test/train split and burn 25% of our data
    print(str(SAMPLE_COUNT*len(Commands)) + " / " + str(SAMPLE_COUNT*len(Commands)) + ": Done!")
    print("\n3: Train Model")
    if(WAIT_TO_CONTINUE): input("Press return to continue...")


    loadingCount = 0
    loadingTotal = len(Commands)*SAMPLE_COUNT

    print("Preprocessing Frequency Bins")
    print("0%    10%    20%    30%    40%    50%    60%    70%    80%    90%   100%")

    for sampleNum in range(len(Dataset)):
        loadingCount = showLoadingBar(loadingCount, loadingTotal)

        preprocess(True, sampleNum)

    print() # So we don't overwrite the loading bar

    # Likely won't need the full dataset any more,
    # so delete to clear up some memory for flying
    del Dataset[:]

    print("Preparing Model")
    print("0%    10%    20%    30%    40%    50%    60%    70%    80%    90%   100%")

    # Shuffle Data
    np.random.shuffle(ProcessedData)
    showLoadingBar(20, 100) # Show that we are about 20% (20/100) done

    # Split Labels and Data
    ProcessedData = np.array(ProcessedData)
    X = ProcessedData[:, :-1]
    y = ProcessedData[:, -1]
    showLoadingBar(40, 100) # Show that we are about 40% (40/100) done

    # Split Test/Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_TRAIN_SPLIT, random_state = 0)
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    showLoadingBar(60, 100) # Show that we are about 60% (60/100) done

    # Actually train it
    LDAModel.fit(X_train, y_train)
    showLoadingBar(100, 100) # Show that we are about 100% (100/100) done

    # print(np.array(ProcessedData))

    print() # So we don't overwrite the loading bar


    randomGuessingPercentage = round(100*(1/len(Commands)),2)
    modelScorePercentage = round(100*LDAModel.score(X_test, y_test),2)

    print("\nPredictions: ", LDAModel.predict(X_train)) # This uses the LDA to predict the label from the given 6D "xTest" value
    print("Real Labels: ", y_train) # These are the actual labels (that LDA has no access to this time)
    print("\nModel successfully classifies " + str(modelScorePercentage) + "% of the samples.")
    print("\nThat is " + str(round(modelScorePercentage - randomGuessingPercentage)) + "% better than randomly guessing at " + str(randomGuessingPercentage) + "% success.\n")


    ProgramStep += 1


def handler(event, sender, data, **args):
    global FlightDataOutput
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        # ALT:  0 | SPD:  0 | BAT: 90 | WIFI: 90 | CAM:  0 | MODE:  6 (example data)
        batLvl = str(data).split("|")[2].lstrip(" ")
        FlightDataOutput[0] = batLvl


def flyDrone():
    global DroneConnected, CommandForDroneThread, FlightDataOutput

    # This is the drone thread
    while(True): # Keep trying to reconnect if connection fails
        drone = tellopy.Tello() ####### Need to find a way to delete this object to reinstatiate it properly ####################


        try:
            # Connect Drone
            drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
            drone.connect()
            drone.wait_for_connection(60.0)


            # Drone Connected! Begin BCI control
            print("\n5: Fly!")
            if(WAIT_TO_CONTINUE): input("Press return to continue...")

            print("\nFlight Dashboard")
            DroneConnected = True


            while(True):
                # If there is a command, execute it then reset
                if(CommandForDroneThread != ""):

                    # Update Dashboard
                    # sys.stdout.write(' %s| Current Command: %s\r' % (FlightDataOutput[0], FlightDataOutput[1]))
                    # sys.stdout.flush()

                    print("\n######### " + str(FlightDataOutput[0]) + "| Current Command: " + str(FlightDataOutput[1]) + "\n")

                    # Execute command
                    updateDrone(drone, CommandForDroneThread, SPEED)
                    CommandForDroneThread = ""

        except Exception as ex:
            print(ex)
        finally:
            drone.quit()

        time.sleep(3)
        if(input("\nDrone connection failed. Try again? (y/n): ").lower() == "n"):
            print("Drone not connected, ending.")
            exit()


def liveClassifier():
    global DatapointBeingCollected, Commands, FlightDataOutput
    # DatapointBeingCollected is the global that stores our current sample

    if(COLLECT_LIVE):
        bandsToPredict = preprocess(False)[:-1]

        DatapointBeingCollected = [[] for i in range(16)] # Empty the currDatapoint

        prediction = LDAModel.predict(np.array(bandsToPredict).reshape(1, -1))[0]

        FlightDataOutput[1] = prediction

        return prediction

    else:
        print("WARNING: Not collecting live data, so a random command will be passed.")
        return Commands[random.randint(0,7)]


def updateDrone(drone, command, speed):
    global DroneOnGround

    # Resets the drone's movement
    # drone.up(0)
    # drone.down(0)
    # drone.forward(0)
    # drone.backward(0)
    # drone.left(0)
    # drone.right(0)
    # drone.clockwise(0)
    # drone.counter_clockwise(0)

    ##TODO Add an if bypass to check if the same argument is passed in twice, reduces jitter?

    if(NO_FLY_MODE):
        # print(command)
        return


    if(command == "Jaw"):
        if(DroneOnGround):
            drone.takeoff()
            DroneOnGround = False
        else:
            drone.land()
            DroneOnGround = True

    elif(command  == "Up"):
        drone.up(speed)
    elif(command == "Down"):
        drone.down(speed)
    elif(command == "Forward"):
        drone.forward(speed)
    elif(command == "Back"):
        drone.backward(speed)
    elif(command == "Left"):
        drone.clockwise(speed)
    elif(command == "Right"):
        drone.counter_clockwise(speed)

    elif(command == "l"): # Not accessable
        drone.left(speed)
    elif(command == "r"): # Not accessable
        drone.right(speed)
    elif(command == "ffr"): # Not accessable
        drone.flip_forwardright()
    elif(command == "lnd"): # Not accessable
        drone.land()
    elif(command == "k"): # Not accessable
        if(input("Are you sure you want to kill the drone? (y/n)") == "y"):
            drone.emergency()


def collectSample(data):
    global Count, DatapointBeingCollected, CommandForDroneThread

    if(not COLLECT_LIVE):
        # pretend to collect a sample, then just don't. liveClassifier()
        # handles randomly submitting a command
        Count = SAMPLE_LEN*2
        time.sleep(1)

    if(Count == 0):
        print("#### START COLLECTING ####") ###################################

    if(Count < SAMPLE_LEN):
        print(Count)
        dataPoint = data.channels_data

        if(Count % 2 == 0): # Halve the data, half is inverted signal (no bueno)
            for point in range(0,len(dataPoint)): ### may need to change 0 to 1, might cause errors ####
                DatapointBeingCollected[point].append(dataPoint[point])
        Count += 1

    else: # Finished collecting sample
        print("#### END COLLECTING ####") ###################################
        if(Count >= SAMPLE_LEN):
            FileNumber[CurrCommand] += 1
            Count = 0
            CommandForDroneThread = liveClassifier()


def collectBaseline(data):
    global Count, SampleFile, FileNumber, Commands, CurrCommand, ProgramStep, Dataset, DatapointBeingCollected

    if(Count < SAMPLE_LEN):
        dataPoint = data.channels_data

        if(Count % 2 == 0): # Halve the data, half is inverted signal (no bueno)
            DatapointBeingCollected[0].append(dataPoint[0]) ###### MIGHT BREAK THINGS ######
            line = str(dataPoint[0])
            for point in range(1,len(dataPoint)):
                # Save the data right away so we don't have to reload and reformat it
                DatapointBeingCollected[point].append(dataPoint[point])
                line += ("," + str(dataPoint[point]))

            if(SampleSaveDirectory != ""):
                SampleFile.write(line + "\n")
                SampleFile.flush()
        Count += 1

        commandPrint = Commands[CurrCommand]
        miniLoadingBar = "#"*(int(np.ceil(Count/25)))
        sys.stdout.write('  %s / %s: %s [%s] \r' % (len(Dataset), SAMPLE_COUNT*len(Commands), commandPrint, miniLoadingBar))
        sys.stdout.flush()

    else: # Finished collecting sample
        if(Count > (SAMPLE_LEN+SAMPLE_RECORDING_TIME*2)): # First run, ignore it
            CurrCommand = random.randint(0,7)
            Count = 0
        else:
            # Only run this code once per loop (when sample count == 250)
            if(Count == SAMPLE_LEN):

                if(SampleSaveDirectory != ""): SampleFile.close()
                FileNumber[CurrCommand] += 1
                Count += 1

                # Save the current sample to the loaded dataset
                DatapointBeingCollected.append(Commands[CurrCommand]) # Slap the label on there
                Dataset.append(DatapointBeingCollected)
                DatapointBeingCollected = [[] for i in range(16)] # Empty the currDatapoint


                ## Check to see if we are done
                if(all(i >= SAMPLE_COUNT for i in FileNumber)):
                    ProgramStep += 1 # Done collecting data, move on to training
                    return


                ## Not done, set up the next sample collection
                CurrCommand = random.randint(0,7)
                while(FileNumber[CurrCommand] >= SAMPLE_COUNT):
                    CurrCommand = random.randint(0,7)


                if(SampleSaveDirectory != ""):
                    SampleFile = open(SampleSaveDirectory + "/" + Commands[CurrCommand] + str(FileNumber[CurrCommand]) + ".txt", 'w')

                commandPrint = Commands[CurrCommand]
                sys.stdout.write('  %s / %s: %s %s \r' % (len(Dataset), SAMPLE_COUNT*len(Commands), commandPrint, " "*20))
                sys.stdout.flush()


            # Pause program to allow for switch
            elif(Count < SAMPLE_LEN + SAMPLE_RECORDING_TIME):
                # To make the program wait for some time, we have to ignore some data
                data.channels_data
                Count += 1

            # Done sample! Next.
            else:
                Count = 0


def loadSavedData():
    """
    Sister function to collectBaseline(). Does essentially the same thing,
    but "collects" prerecorded data instead.
    """
    global Dataset, ProcessedData, Commands, Count, ProgramStep
    loadingCount = 0
    loadingTotal = len(Commands)*SAMPLE_COUNT
    loadingBar = ""

    print("\n1: Connect BCI")
    print("Not Connecting a BCI. Done.\n")
    print("2: Collect Data")
    if(WAIT_TO_CONTINUE): input("Presss return to continue...")

    print("Loading Data")
    print("0%    10%    20%    30%    40%    50%    60%    70%    80%    90%   100%")

    for command in Commands:
        for sampleNum in range(0, SAMPLE_COUNT):

            ## Loading Bar
            loadingCount += 1
            loadingBar = "#"*int(np.ceil(70*loadingCount/loadingTotal))
            sys.stdout.write('[%s] \r' % (loadingBar))
            sys.stdout.flush()
            ## Loading Bar


            channels = [[] for i in range(16)]

            with open(SampleSaveDirectory + "/" + command + str(sampleNum) + ".txt", "r") as file:

                for line in file:
                    if(Count % 2 == 0):
                        line = line.lstrip("[")
                        line = line.strip("\n")
                        line = line.strip("'")
                        line = line.rstrip("]")
                        line = line.replace(" ", "")

                        # Verifies that Git hasn't messed up the data
                        if line == "<<<<<<<HEAD":
                            print("Encountered an error with unresolved GitHub Merge Conflicts.")
                            print("Problem file: ", command + str(sampleNum) + ".txt")

                        line = line.split(",")

                        for i in range(len(line)):
                            channels[i].append(float(line[i]))
                        # labelToAppend = line[-1].strip("'")

                    Count += 1
                channels.append(command) # Add the label
                Dataset.append(channels)

    print() # So we don't overwrite the loading bar
    ProgramStep += 1


def bciStream(data):
    """
    A 'main' function of sorts, the OpenBCICyton library calls this funtion
    on a while(True) loop when the BCI is connected and feeding data.
    This will act as the hub for our program.
    """
    global ProgramStep, FirstRun, Count

    if(ProgramStep == 0):
        if(COLLECT_LIVE): checkForRailedChannels(data)
        else:             ProgramStep+=1

    elif(ProgramStep == 1):
        if(COLLECT_LIVE and not SkipCollecting): collectBaseline(data)
        else:                                    loadSavedData()

    elif(ProgramStep == 2):
        trainModel()

    else:
        if(FirstRun):
            print("\n4: Connect Drone")
            if(WAIT_TO_CONTINUE): input("Press return to continue...")

            droneThread = threading.Thread(target=flyDrone)
            droneThread.start()
            FirstRun = False
            Count = 0

        if(DroneConnected):
            collectSample(data)
        else:
            # Keep the queue empty until the drone connects
            if(COLLECT_LIVE): data.channels_data





# Start everything up
print("\nNAT Brain Drone Controller")
print("1: Connect BCI")
print("2: Collect Data")
print("3: Train Model")
print("4: Connect Drone")
print("5: Fly!")

if(WAIT_TO_CONTINUE): input("\nPress return to continue...")

if(COLLECT_LIVE):
    print("\n1: Connect BCI")
    board = OpenBCICyton(port='/dev/tty.usbserial-DM01N7JO', daisy=True)
    board.start_stream(bciStream)

else:
    # Don't bother connecting board, just run the bciStream() like OpenBCICyton would
    while(True):
        bciStream("")
