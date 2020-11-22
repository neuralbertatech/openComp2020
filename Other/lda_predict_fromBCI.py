"""
Reads live data from the OpenBCI and uses the LDA model saved as "lda_model.pk"
to make predictions.

This is an old implementation, the complete implementation can be found in
masterController.py
"""


from pyOpenBCI import OpenBCICyton
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

SAMPLE_LEN = 250

lda_model = pickle.load(open('lda_model.pk', 'rb'))
CurrentSample = []
DataCounter = 0
HalfData = True
plotData = False    # Save the data plots
showFFTBins = True  # Draw FFT bin lines on the plots


def collectSample(bciData):
    global CurrentSample, DataCounter, HalfData

    ### Add error message for railed channels

    # Properly implemented in masterController.py

    if(HalfData and DataCounter%2 == 0):
        CurrentSample.append(bciData.channels_data)
    elif(not HalfData):
        CurrentSample.append(bciData.channels_data)
    DataCounter += 1


def predictSample():
    global CurrentSample, DataCounter

    # Bin the sample
    # [Delta (0-4), Theta (4-7.5), Alpha (7.5-12.5), Beta (12.5-30), Gamma (30-70)]
    bands = []

    for channelNum in range(16):
        # Appy a Savgol filter to smooth out imperfect data
        smoothedData = savgol_filter(np.array(CurrentSample)[:,channelNum], 11, 2)
        x = np.arange(0,int(np.ceil(SAMPLE_LEN/2)),1)

        if(plotData):
            plt.clf()
            plt.plot(x, smoothedData, "r")
            plt.savefig("dataplt" + str(channelNum) + ".png")



        # FFT the data
        sp = np.fft.fft(smoothedData)
        freq = np.fft.fftfreq(x.shape[-1], 1/SAMPLE_LEN)

        freq = freq[1:int(np.ceil(SAMPLE_LEN/4))] # Only Care about positive
        sp = sp[1:int(np.ceil(SAMPLE_LEN/4))]
        sp = np.sqrt(sp.real**2 + sp.imag**2)


        if(plotData):
            plt.clf()
            plt.plot(freq, sp, "r")
            if(showFFTBins):
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
    bandsToPredict = []
    for bandNum in range(5):
        bandsToPredict.append(round(np.average(np.array(bands)[:,bandNum]),2))



    # Now Predict!
    command = lda_model.predict(np.array(bandsToPredict).reshape(1, -1))[0]

    DataCounter = 0
    CurrentSample = []

    return command


def predict(bciData):
    global CurrentSample, DataCounter


    if(DataCounter < SAMPLE_LEN): #250 samples = 1 second (250Hz)
        collectSample(bciData)
    else:
        print(predictSample())


board = OpenBCICyton(port='/dev/tty.usbserial-DM01N7JO', daisy=True)
board.start_stream(predict)
