import os
import re

def main():
    fileNames = []

    for i in range(0, 680):
        fileNames.append("Back" + str(i) + ".txt")
        fileNames.append("Down" + str(i) + ".txt")
        fileNames.append("Forward" + str(i) + ".txt")
        fileNames.append("Left" + str(i) + ".txt")
        fileNames.append("Right" + str(i) + ".txt")
        fileNames.append("Stay" + str(i) + ".txt")
        fileNames.append("Up" + str(i) + ".txt")
        fileNames.append("Jaw" + str(i) + ".txt")

    header = 'channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8, channel9, channel10, channel11, channel12, channel13, channel14, channel15, channel16, direction\n'
    csv = open("dataCSV.csv", "w+")
    csv.write(header)

    for i in range(0, len(fileNames)):
        file = open("./TrainingData/CameronLimbs/" + fileNames[i], "r")
        lines = file.readlines()
        for line in lines:
            line = line.lstrip("[")
            line = line.strip("\n")
            line = line.strip("'")
            line = line.rstrip("]")
            line = line.replace(" ", "")

            # Verifies that Git hasn't messed up the data
            if line == "<<<<<<<HEAD":
                print("Problem file: ", fileNames[i])

            fname = os.path.basename(file.name).rstrip(".txt")
            fname = re.sub(r'\d+', '', fname)

            csv.write(line + "\n")



if __name__ == "__main__":
    main()
