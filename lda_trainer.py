import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Define the data
commands = ["Back", "Down", "Forward", "Left", "Right", "Stay", "Up", "Jaw"]
dataset = []
totalSamples = 680
sampleLen = 250
halfData = True # Sometimes every second point of BCI data is inverted. Remove?
savePlots = False # True drastically increases runtime

showLoadingBar = True
loadingBar = ""
loadingCount = 0
loadingTotal = len(commands)*totalSamples

plotColors = ["#ff0000", "#ff0038", "#ff0060", "#ff0085", "#ff26a9", "#ff4aca", "#ee65e7", "#d47bff", "#b68fff", "#92a0ff", "#6aaeff", "#38bbff", "#00c5ff", "#00ceff", "#00d6ff", "#00dcff"]


### Load & Format the data ###
if(showLoadingBar):
    print("Loading Data")
    print("0%    10%    20%    30%    40%    50%    60%    70%    80%    90%   100%") # 72 loading points

for command in commands:
    for sampleNum in range(0, totalSamples):
        if(showLoadingBar):
            loadingCount += 1
            loadingBar = "#"*int(np.ceil(70*loadingCount/loadingTotal))
            sys.stdout.write('[%s] \r' % (loadingBar))
            sys.stdout.flush()


        channels = [[] for i in range(16)]

        with open("./TrainingData/CameronLimbs/" + command + str(sampleNum) + ".txt", "r") as file:
            count = 0
            labelToAppend = ""
            for line in file:
                if(halfData and count%2 == 0):
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

                    for i in range(len(line)-1):
                        channels[i].append(float(line[i]))
                    labelToAppend = line[-1].strip("'")

                elif(not halfData):
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

                    for i in range(len(line)-1):
                        channels[i].append(float(line[i]))
                    labelToAppend = line[-1].strip("'")

                count += 1
            channels.append(labelToAppend) # Add the label
            dataset.append(channels)
### End Load & Format the data ###



### Extract the Frequency Bins ###
if(showLoadingBar):
    loadingBar = ""
    loadingCount = 0
    print("\nDone Loading Data!\n")
    print("Extracting Frequency Bins")
    print("0%    10%    20%    30%    40%    50%    60%    70%    80%    90%   100%") # 72 loading points

binnedDataset = []

for sampleNum in range(len(dataset)):

    if(showLoadingBar):
        loadingCount += 1
        loadingBar = "#"*int(np.ceil(70*loadingCount/loadingTotal))
        sys.stdout.write('[%s] \r' % (loadingBar))
        sys.stdout.flush()

    # [Delta (0-4), Theta (4-7.5), Alpha (7.5-12.5), Beta (12.5-30), Gamma (30-70)]
    bands = []

    for channelNum in range(16):
        # Appy a Savgol filter to smooth out imperfect data
        smoothedData = savgol_filter(dataset[sampleNum][channelNum], 11, 2)
        x = np.arange(0,125,1)


        if(savePlots):
            plt.clf()
            plt.plot(x, smoothedData, "r")
            plt.savefig("dataplt" + str(channelNum) + ".png")



        # FFT the data
        sp = np.fft.fft(smoothedData)
        freq = np.fft.fftfreq(x.shape[-1], 1/sampleLen)

        freq = freq[1:int(np.ceil(sampleLen/4))] # Only Care about positive
        sp = sp[1:int(np.ceil(sampleLen/4))]
        sp = np.sqrt(sp.real**2 + sp.imag**2)



        if(savePlots):
            plt.clf()
            plt.plot(freq, sp, plotColors[channelNum])
            # plt.axvline(x=4, color="k")
            # plt.axvline(x=7.5, color="k")
            # plt.axvline(x=12.5, color="k")
            # plt.axvline(x=30, color="k")
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

    # print(np.round(bands, 2))


    # Add the label back
    avgBands.append(dataset[sampleNum][-1])
    binnedDataset.append(avgBands)
### End Extract the Frequency Bins ###



### Prepare Model ###
if(showLoadingBar):
    print("\nDone Extraxting!\n")
    print("Preparing Model...")

# Shuffle Data
np.random.shuffle(binnedDataset)

# Split Labels and Data
binnedDataset = np.array(binnedDataset)
X = binnedDataset[:, :-1]
y = binnedDataset[:, -1]

# Split Test/Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

### End Prepare Model ###



### Train Model ###
if(showLoadingBar):
    print("Training Model...")

lda_model = LinearDiscriminantAnalysis()
lda_model.fit(X_train, y_train)
### End Train Model ###



### Test Prediction Accuracy ###
if(showLoadingBar):
    print("Scoring Model...\n")

# print("Predictions: ", lda_model.predict(X_train)) # This uses the LDA to predict the label from the given 6D "xTest" value
# print("Real Labels: ", y_train) # These are the actual labels (that LDA has no access to this time)


randomGuessingPercentage = round(100*(1/len(commands)),2)
modelScorePercentage = round(100*lda_model.score(X_test, y_test),2)

print("Model successfully classifies " + str(modelScorePercentage) + "% of the samples.")
print("That is " + str(round(modelScorePercentage - randomGuessingPercentage)) + "% better than randomly guessing at " + str(randomGuessingPercentage) + "% success.")
### End Test Prediction Accuracy ###



### Save Model ###
if(showLoadingBar):
    print("\nSaving Model...")

pickle.dump(lda_model, open('lda_model.pk','wb'))

if(showLoadingBar):
    print("Done!\n\n")
### Save Model ###
