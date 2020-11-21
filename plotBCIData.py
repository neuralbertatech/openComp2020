"""
This program will plot your recorded brain data.

### NOTE ###
This program is a bigger memory hog than Chrome!
If you are generating a gif with many frames, you may run into a problem
where your OS restricts the number of open files. To circumvent this, use
"ulimit -n 700" or some value slightly larger than the number of frames
you intent to plot.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import savgol_filter

queueToPlot = "Right"
startNumber = 0
numberToPlot = 680
loadingCount = 0
GeneratePlots = True
GenerateGif = True # Must have plots generated already

if(GeneratePlots):
    print("Generating Plots")

    for sampleNum in range(startNumber, startNumber+numberToPlot):

        loadingCount += 1
        sys.stdout.write('Generating %s / %s (%s%s)\r' % (loadingCount, numberToPlot-startNumber, round(100*loadingCount/(numberToPlot-startNumber), 2),"%"))
        sys.stdout.flush()



        channels = [[] for i in range(16)]

        with open("TrainingData/CameronLimbs/" + queueToPlot + str(sampleNum) + ".txt", "r") as file:

            # Format data
            for line in file:
                #Remove "[]" and convert to an array
                line = line[1:-2].split(", ")
                line.pop()

                # Convert strings to ints and convert to channel-wise data
                for i in range(len(line)):
                    channels[i].append(int(line[i]))


        # Plot the Data
        x = np.linspace(0,len(channels[0])-1, len(channels[0]))


        ### Only plot evert second point ###
        plotEverySecondChannel = True
        if(plotEverySecondChannel):
            x = np.linspace(0,len(channels[0])-1, int(np.ceil(len(channels[0])/2)))
            for channel in range(len(channels)):
                temp = []
                for i in range(len(channels[channel])):
                    if i%2 == 0:
                        temp.append(channels[channel][i])
                channels[channel] = temp
        ### Only plot every second point ###



        plotColors = ["#ff0000", "#ff0038", "#ff0060", "#ff0085", "#ff26a9", "#ff4aca", "#ee65e7", "#d47bff", "#b68fff", "#92a0ff", "#6aaeff", "#38bbff", "#00c5ff", "#00ceff", "#00d6ff", "#00dcff"]


        # Individual Images
        for channel in range(len(channels)):
            ### SAVGOL FILTER PLOT ###
            smoothedData = savgol_filter(channels[channel], 11, 2)
            plt.clf()
            plt.plot(x, smoothedData, plotColors[channel])
            ### SAVGOL FILTER PLOT ###

            # plt.clf()
            # plt.plot(x, channels[channel], plotColors[channel])


            plt.xlabel("Sample")
            plt.ylabel("μV")

            plt.savefig("BrainDataPlot/" + str(channel) + ".png")

        # Combine
        images = [Image.open("BrainDataPlot/" + x) for x in ['0.png', '1.png', '2.png', '3.png', '4.png', '5.png', '6.png', '7.png', '8.png', '9.png', '10.png', '11.png', '12.png', '13.png', '14.png', '15.png']]
        widths, heights = zip(*(i.size for i in images))

        new_im = Image.new('RGB', (widths[0]*4, heights[0]*4))

        x_offset = 0
        y_offset = 0
        count = 0
        for im in images:
            if(count >= 4):
                count = 0
                y_offset += heights[0]
                x_offset = 0

            new_im.paste(im, (x_offset,y_offset))
            x_offset += widths[0]
            count += 1

        new_im.save('BrainDataPlot/' + queueToPlot + '/' + queueToPlot + str(sampleNum) + '.jpg')
    print("\nDone Generating Plots!")


if(GenerateGif):
    # Make Final .gif
    print("Generating final gif...")


    imagesToOpen = []
    for sampleNum in range(startNumber, startNumber+numberToPlot):
        imagesToOpen.append('BrainDataPlot/' + queueToPlot + '/' + queueToPlot + str(sampleNum) + '.jpg')

    gifImages = [Image.open(y) for y in imagesToOpen]
    gifWidths, gifHeights = zip(*(i.size for i in gifImages))

    print("Images opened, compiling...")

    gifImages[0].save('BrainDataPlot/' + queueToPlot + '/Animation.gif', save_all=True, append_images=gifImages[1:], loop=0, fps=24)


print("Done!")
