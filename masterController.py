"""
This is the entirety of the python brain drone controller. All methods are contained within this file.
To use the brain drone, simply type:
"python masterController.py"
into the command line and follow the onscreen instructions.

If you have collected a baseline, but wish to rerun the program wihtout recollecting it, simply call:
"python masterController.py nocol"

If your baseline files are stored in a directory different than the global variable SampleSaveDirectory,
then include your directory (relative to this file) as an argument, like:
"python masterController.py nocol dataDirectory"
"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_validate
from scipy.signal import savgol_filter
from pyOpenBCI import OpenBCICyton
import matplotlib.pyplot as plt
import numpy as np
import threading
import tellopy
import random
import select
import time
import sys
import os

SAMPLE_LEN = 250                 # 250samples = 1s
SAMPLE_COUNT = 10                # Amount of samples of each command to record
SAMPLE_RECORDING_TIME = 250      # Amount of samples to switch (time between recordings in units of 1/250s), (minimum=1)
NUM_FOLDS = 10                   # For k-fold cross validation
SAVE_PLOTS = False               # Plot data while collecting
WAIT_TO_CONTINUE = False         # False will automatically skip program pauses
CONNECT_DRONE = False            # False will simulate drone responses
COLLECT_LIVE = False             # False will generate random data (reccomended to enable NO_FLY_MODE)
NO_FLY_MODE = False              # Connect drone, except instead of flying, just print output
SPEED = 20                       # value 0-100 that controls the drones speed

LDAAVGModel = LinearDiscriminantAnalysis()
LDAFFTModel = LinearDiscriminantAnalysis()
LDARAWModel = LinearDiscriminantAnalysis()

Dataset = []
ProcessedData = []
ProcessedDataFFT = []
ProcessedDataRAW = []
DatapointBeingCollected = [[] for i in range(16)]

FlightDataOutput = ["", "", "", "", ""]

FileNumber = [-1, 0, 0, 0, 0, 0, 0, 0] # For some reason it just doesn't record the first Command, so start at -1
Commands = ["Up", "Down", "Forward", 'Back', "Left", "Right", "Stay", "Land"]
CommandForDroneThread = ""
CurrCommand = 0
ProgramStep = 0
FirstRun = True
DroneConnected = False
Count = SAMPLE_LEN+SAMPLE_RECORDING_TIME*2 + 1  # Used throughout code to keep a global count of iterations, set to large num to skip first iter of collecting

SampleSaveDirectory = "TrainingData/masterControllerSessions"  # Set to "" if you do not wish to record samples
if(SampleSaveDirectory != ""):
    SampleFile = open(SampleSaveDirectory + "/" + Commands[CurrCommand] + str(FileNumber[CurrCommand]) + ".txt", 'w')

# NOTE: COLLECT_LIVE should be set to True if we choose to reuse data. COLLECT_LIVE = False will randomly control the drone.
SkipCollecting = False
if(len(sys.argv) > 1):
    if(str(sys.argv[1]) == "nocol"):
        if(SampleSaveDirectory != ""):
            SkipCollecting = True
        else:
            if(len(sys.argv) > 2):
                SampleSaveDirectory = str(sys.argv[2])
            else:
                print("No sample directory defined in code, please include the data directory as a second argumet like 'python masterController.py nocol dataDirectory' to utilize prerecorded samples.\n")
                exit()
    else:
        print("Unknown argument '" + str(sys.argv[1]) + "'\n")
        exit()

if(not COLLECT_LIVE and not NO_FLY_MODE and CONNECT_DRONE):
    print("\n#############################################################################")
    print("# WARNING: Drone is configured to fly and is being fed random commands.     #")
    print("#          It is strongly recommended to enable NO_FLY_MODE.                #")
    print("#############################################################################")


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

    channelLen = 0
    if(training): channelLen = len(Dataset[sampleNum][0])
    else: channelLen = len(DatapointBeingCollected[0])

    avgChannel = [0 for i in range(channelLen)]

    for channelNum in range(16):

        # Appy a Savgol filter to smooth out imperfect data
        if(training):
            smoothedData = savgol_filter(Dataset[sampleNum][channelNum], 11, 2)
        else:
            smoothedData = savgol_filter(DatapointBeingCollected[channelNum], 11, 2)
        x = np.arange(0,125,1)

        # Plot
        if(SAVE_PLOTS and training): # We never want to plot while flying
            plt.clf()
            plt.plot(x, smoothedData, "r")
            plt.savefig("savgol" + str(channelNum) + ".png")

        # Create Avg channel
        for dataPoint in range(len(smoothedData)):
            avgChannel[dataPoint] += smoothedData[dataPoint]

        # Plot
        if(SAVE_PLOTS and training and channelNum==15): # We never want to plot while flying
            plt.clf()
            plt.plot(x, avgChannel, "b")
            plt.savefig("avgChannel" + str(channelNum) + ".png")

    if(training):
        # Add the label back
        avgChannel.append(Dataset[sampleNum][-1])
        ProcessedData.append(avgChannel)

    else:
        avgChannel.append("")
        return avgChannel



def preprocessRAW(training: bool, sampleNum = 0):
    """
    This func takes in a single data point and preprocesses it.
    This will work for both training and flying, so it needs to be fast and flexible
    - training tells us if we are referencing the dataset or classifying
    - sampleNum tells us which iter we are training
    """
    global Dataset, ProcessedDataRAW, Commands, DatapointBeingCollected

    channelLen = 0
    if(training): channelLen = len(Dataset[sampleNum][0])
    else: channelLen = len(DatapointBeingCollected[0])

    bigChannel = []

    for channelNum in range(16):

        # Appy a Savgol filter to smooth out imperfect data
        if(training):
            smoothedData = savgol_filter(Dataset[sampleNum][channelNum], 11, 2)
        else:
            smoothedData = savgol_filter(DatapointBeingCollected[channelNum], 11, 2)
        x = np.arange(0,125,1)

        # Add data to one big channel
        for dataPoint in range(len(smoothedData)):
            bigChannel.append(smoothedData[dataPoint])


    if(training):
        # Add the label back
        bigChannel.append(Dataset[sampleNum][-1])
        ProcessedDataRAW.append(bigChannel)
    else:
        bigChannel.append("")
        return bigChannel


def preprocessFFT(training: bool, sampleNum = 0):
    """
    This func takes in a single data point and preprocesses it.
    This will work for both training and flying, so it needs to be fast and flexible
    - training tells us if we are referencing the dataset or classifying
    - sampleNum tells us which iter we are training
    """
    global Dataset, ProcessedDataFFT, Commands, DatapointBeingCollected

    # [Delta (0-4), Theta (4-7.5), Alpha (7.5-12.5), Beta (12.5-30), Gamma (30-70)]
    bands = []

    for channelNum in range(16):

        # Appy a Savgol filter to smooth out imperfect data
        if(training):
            smoothedData = savgol_filter(Dataset[sampleNum][channelNum], 11, 2)
        else:
            smoothedData = savgol_filter(DatapointBeingCollected[channelNum], 11, 2)
        x = np.arange(0,125,1)


        # Plot
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


        # Plot
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

    # Now, cast the set of bins of each electrode into a single set of bins (average)
    avgBands = []
    for bandNum in range(5):
        avgBands.append(round(np.average(np.array(bands)[:,bandNum]),2))

    if(training):
        # Add the label back
        avgBands.append(Dataset[sampleNum][-1])
        ProcessedDataFFT.append(avgBands)
    else:
        avgBands.append("")
        return avgBands


def checkForRailedChannels(data):
    global ProgramStep, Count, DatapointBeingCollected, CommandForDroneThread
    # print status of all channels (Railed, Good)

    if(not COLLECT_LIVE):
        allChannelsGood = True

    if(Count < SAMPLE_LEN):
        dataPoint = data.channels_data

        if(Count % 2 == 0): # Halve the data, half is inverted signal (no bueno)
            for point in range(0,len(dataPoint)):
                DatapointBeingCollected[point].append(dataPoint[point])
        Count += 1

    else: # Finished collecting sample
        railedChannels = []

        # Check for rails
        for channelNum in range(16):
            thisLineGood = False

            line = DatapointBeingCollected[channelNum]
            value = line[0]
            for i in line:
                if(i != value):
                    thisLineGood = True
                    break

            if(not thisLineGood):
                railedChannels.append(channelNum)

        # Reset
        DatapointBeingCollected = [[] for i in range(16)]
        Count = 0

        # Decide if we are good to go
        if(len(railedChannels) == 0):
            allChannelsGood = True
        else:
            print("Railed Channels: ", railedChannels)


        if(allChannelsGood):
            print("\nNo railed channels, begin collecting data")
            print("\n2: Collect Data")
            if(WAIT_TO_CONTINUE): input("Press return to continue...")
            ProgramStep += 1


def trainModel():
    global ProgramStep, ProcessedData, ProcessedDataFFT, ProcessedDataRAW, LDAAVGModel, Commands

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
        preprocessFFT(True, sampleNum)
        preprocessRAW(True, sampleNum)

    print() # So we don't overwrite the loading bar

    # Likely won't need the full dataset any more,
    # so delete to clear up some memory for flying
    del Dataset[:]

    #########################################################################################################
    print("Preparing AVG Model")
    print("0%    10%    20%    30%    40%    50%    60%    70%    80%    90%   100%")

    # Shuffle Data
    np.random.shuffle(ProcessedData)
    showLoadingBar(33, 100) # Show that we are about 33% (33/100) done

    # Split Labels and Data
    ProcessedData = np.array(ProcessedData)
    Xavg = ProcessedData[:, :-1].astype(np.float64)
    yavg = ProcessedData[:, -1]
    showLoadingBar(66, 100) # Show that we are about 66% (66/100) done

    # Actually train it
    LDAAVGModel.fit(Xavg, yavg)
    showLoadingBar(100, 100) # Show that we are about 100% (100/100) done

    print() # So we don't overwrite the loading bar
    #########################################################################################################

    #########################################################################################################
    print("Preparing FFT Model")
    print("0%    10%    20%    30%    40%    50%    60%    70%    80%    90%   100%")

    # Shuffle Data
    np.random.shuffle(ProcessedDataFFT)
    showLoadingBar(33, 100) # Show that we are about 33% (33/100) done

    # Split Labels and Data
    ProcessedDataFFT = np.array(ProcessedDataFFT)
    Xfft = ProcessedDataFFT[:, :-1].astype(np.float64)
    yfft = ProcessedDataFFT[:, -1]
    showLoadingBar(66, 100) # Show that we are about 66% (66/100) done

    # Actually train it
    LDAFFTModel.fit(Xfft, yfft)
    showLoadingBar(100, 100) # Show that we are about 100% (100/100) done

    print() # So we don't overwrite the loading bar
    #########################################################################################################

    #########################################################################################################
    print("Preparing RAW Model")
    print("0%    10%    20%    30%    40%    50%    60%    70%    80%    90%   100%")

    # Shuffle Data
    np.random.shuffle(ProcessedDataRAW)
    showLoadingBar(33, 100) # Show that we are about 33% (33/100) done

    # Split Labels and Data
    ProcessedDataRAW = np.array(ProcessedDataRAW)
    Xraw = ProcessedDataRAW[:, :-1].astype(np.float64)
    yraw = ProcessedDataRAW[:, -1]
    showLoadingBar(66, 100) # Show that we are about 66% (66/100) done

    # Actually train it
    LDARAWModel.fit(Xraw, yraw)
    showLoadingBar(100, 100) # Show that we are about 100% (100/100) done

    print() # So we don't overwrite the loading bar
    #########################################################################################################

    scores = cross_validate(LDAAVGModel, Xavg, yavg, cv=NUM_FOLDS)["test_score"]
    modelScorePercentage = round(100*(sum(scores)/len(scores)),2)
    print("\nAVG Model successfully classifies " + str(modelScorePercentage) + "% of the samples.")

    scores = cross_validate(LDAFFTModel, Xfft, yfft, cv=NUM_FOLDS)["test_score"]
    modelScorePercentage = round(100*(sum(scores)/len(scores)),2)
    print("FFT Model successfully classifies " + str(modelScorePercentage) + "% of the samples.")

    scores = cross_validate(LDARAWModel, Xraw, yraw, cv=NUM_FOLDS)["test_score"]
    modelScorePercentage = round(100*(sum(scores)/len(scores)),2)
    print("RAW Model successfully classifies " + str(modelScorePercentage) + "% of the samples.")

    print("Randomly guessing successfully classifies " + str(round(100*(1/len(Commands)),2)) + "% of the time.")

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

    try:
        # Connect Drone
        if(CONNECT_DRONE):
            drone = tellopy.Tello()
            drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
            drone.connect()
            drone.wait_for_connection(60.0)


        # Drone Connected! Begin BCI control
        print("\n5: Fly!")
        if(not COLLECT_LIVE and not CONNECT_DRONE): print("WARNING:   BCI data and drone response is simulated.")
        elif(not COLLECT_LIVE): print("WARNING:   BCI data is simulated.")
        elif(not CONNECT_DRONE): print("WARNING:   Drone response is simulated.")
        print("IMPORTANT: To safely land the drone, press return at any time during flight.")
        print("           To immediately kill the drone, press k then return.")
        if(WAIT_TO_CONTINUE): input("Press return to continue...")

        print("\nFlight Dashboard")
        DroneConnected = True

        while(True):
            # If there is a command, execute it then reset
            if(CommandForDroneThread != ""):

                #### Kill drone ####
                sysInput = select.select([sys.stdin], [], [], 1)[0]
                if(sysInput):
                    value = sys.stdin.readline().rstrip()
                    if(value == "k"):
                        print("\n#################################")
                        print("### Killing drone immediately ###")
                        print("#################################\n")
                        if(CONNECT_DRONE): updateDrone(drone, "k", SPEED)
                        break

                    elif(value == ""):
                        print("\n###################################")
                        print("### Safely landing the drone... ###")
                        print("###################################\n")
                        if(CONNECT_DRONE): updateDrone(drone, "lnd", SPEED)
                        break
                #### Kill drone ####

                # Keep spacing consistent so the dashboard is pretty
                output = [""] # Put one item (bat) in it already so we can keep the indicies the same
                for i in range(1,5):
                    space = " "*(7 - len(FlightDataOutput[i]))
                    output.append(str(FlightDataOutput[i]) + space)
                # Keep spacing consistent so the dashboard is pretty

                if(not CONNECT_DRONE): FlightDataOutput[0] = "BAT: 100 " # If drone not connected, handler won't be updating bat, simulate.

                print("### " + str(FlightDataOutput[0]) + "| AVG: " + output[2] + " | RAW: " + output[3] + " | FFT: " + output[4] + " | Command: " + output[1] + " ###")


                # Execute command
                if(CONNECT_DRONE): updateDrone(drone, CommandForDroneThread, SPEED)
                CommandForDroneThread = ""

                time.sleep(0.9)

            # Wait for a hot minute so we don't just destroy the CPU with this thread
            time.sleep(0.1)

    except Exception as ex:
        print(ex)

    finally:
        if(CONNECT_DRONE): drone.quit()


    time.sleep(3)
    print("Drone failed to connect, ending.")
    exit()


def liveClassifier():
    global DatapointBeingCollected, Commands, FlightDataOutput
    # DatapointBeingCollected is the global that stores our current sample

    if(COLLECT_LIVE):
        bandsToPredict = preprocess(False)[:-1]
        bandsToPredictFFT = preprocessFFT(False)[:-1]
        bandsToPredictRAW = preprocessRAW(False)[:-1]

        DatapointBeingCollected = [[] for i in range(16)] # Empty the currDatapoint

        FlightDataOutput[2] = LDAAVGModel.predict(np.array(bandsToPredict).reshape(1, -1))[0]
        FlightDataOutput[3] = LDAFFTModel.predict(np.array(bandsToPredictFFT).reshape(1, -1))[0]
        FlightDataOutput[4] = LDARAWModel.predict(np.array(bandsToPredictRAW).reshape(1, -1))[0]

    else: # Not collecting live, just do random.
        FlightDataOutput[2] = Commands[random.randint(0,len(Commands)-1)]
        FlightDataOutput[3] = Commands[random.randint(0,len(Commands)-1)]
        FlightDataOutput[4] = Commands[random.randint(0,len(Commands)-1)]


    # If two of the commands match, great! If not, stay.
    if(FlightDataOutput[2] == FlightDataOutput[3]):
        FlightDataOutput[1] = FlightDataOutput[2]
    elif (FlightDataOutput[3] == FlightDataOutput[4]):
        FlightDataOutput[1] = FlightDataOutput[3]
    elif (FlightDataOutput[2] == FlightDataOutput[4]):
        FlightDataOutput[1] = FlightDataOutput[4]
    else:
        FlightDataOutput[1] = "Stay"


    # AVG is really good at predicting Jaw, it has 100% weight.
    if(FlightDataOutput[2] == "Jaw"):
        FlightDataOutput[1] = "Jaw"


    # Send the final result
    return FlightDataOutput[1]



def updateDrone(drone, command, speed):
    global DroneOnGround

    # Resets the drone's movement
    drone.up(0)
    drone.down(0)
    drone.forward(0)
    drone.backward(0)
    drone.left(0)
    drone.right(0)
    drone.clockwise(0)
    drone.counter_clockwise(0)

    if(NO_FLY_MODE):
        return

    if(command == "Land"):
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
    elif(command == "lnd"):
        drone.land()
    elif(command == "k"):
        drone.emergency()

    elif(command == "l"): # Not accessible
        drone.left(speed)
    elif(command == "r"): # Not accessible
        drone.right(speed)
    elif(command == "ffr"): # Not accessible
        drone.flip_forwardright()


def collectSample(data):
    global Count, DatapointBeingCollected, CommandForDroneThread

    if(not COLLECT_LIVE):
        # pretend to collect a sample, then just don't. liveClassifier()
        # handles randomly submitting a command
        Count = SAMPLE_LEN*2
        time.sleep(1)

    if(Count < SAMPLE_LEN):
        dataPoint = data.channels_data

        if(Count % 2 == 0): # Halve the data, half is inverted signal (no bueno)
            for point in range(0,len(dataPoint)): ### may need to change 0 to 1, might cause errors ####
                DatapointBeingCollected[point].append(dataPoint[point])
        Count += 1

    else: # Finished collecting sample
        if(Count >= SAMPLE_LEN):
            FileNumber[CurrCommand] += 1
            Count = 0
            CommandForDroneThread = liveClassifier()


def collectBaseline(data):
    global Count, SampleFile, FileNumber, Commands, CurrCommand, ProgramStep, Dataset, DatapointBeingCollected

    if(Count < SAMPLE_LEN):
        dataPoint = data.channels_data

        if(Count % 2 == 0): # Halve the data, half is inverted signal (no bueno)
            DatapointBeingCollected[0].append(dataPoint[0])
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
            CurrCommand = random.randint(0,len(Commands)-1)
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
                CurrCommand = random.randint(0,len(Commands)-1)
                while(FileNumber[CurrCommand] >= SAMPLE_COUNT):
                    CurrCommand = random.randint(0,len(Commands)-1)


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


    print("\n1: Connect BCI")
    print("Not connecting BCI, data will be simulated.\n")
    print("2: Collect Data")
    if(WAIT_TO_CONTINUE): input("Presss return to continue...")

    print("Loading Data")
    print("0%    10%    20%    30%    40%    50%    60%    70%    80%    90%   100%")

    for command in Commands:
        for sampleNum in range(0, SAMPLE_COUNT):
            loadingCount = showLoadingBar(loadingCount, loadingTotal)

            channels = [[] for i in range(16)]

            with open(SampleSaveDirectory + "/" + command + str(sampleNum) + ".txt", "r") as file:

                for line in file:
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
    global ProgramStep, FirstRun, Count, DroneConnected

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
            if(not CONNECT_DRONE): print("Not connecting drone, data will be simulated.")

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

else: # Fake the data
    # Don't bother connecting board, just run the bciStream() like OpenBCICyton would
    while(True):
        bciStream("")
