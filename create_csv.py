import os
import re

def main():
    fileNames = ['Back0.txt', 'Back1.txt', 'Back2.txt', 'Back3.txt', 'Back4.txt', 'Back5.txt', 'Back6.txt', 'Back7.txt', 'Back8.txt', 
    'Back9.txt', 'Back10.txt', 'Back11.txt', 'Back12.txt', 'Back13.txt', 'Back14.txt', 
    'Down0.txt', 'Down1.txt', 'Down2.txt', 'Down3.txt', 'Down4.txt', 'Down5.txt', 'Down6.txt', 'Down7.txt', 'Down8.txt', 
    'Down10.txt', 'Down11.txt', 'Down12.txt', 'Down13.txt', 'Down14.txt', 'Down15.txt', 
    'Forward0.txt', 'Forward1.txt', 'Forward2.txt', 'Forward3.txt', 'Forward4.txt', 'Forward5.txt', 'Forward6.txt', 'Forward7.txt', 
    'Forward8.txt', 'Forward9.txt', 'Forward10.txt', 'Forward11.txt', 'Forward12.txt', 'Forward13.txt', 'Forward14.txt', 'Forward15.txt', 
    'Left0.txt', 'Left1.txt', 'Left2.txt', 'Left3.txt', 'Left4.txt', 'Left5.txt', 'Left6.txt', 'Left7.txt', 
    'Left8.txt', 'Left9.txt', 'Left10.txt', 'Left11.txt', 'Left12.txt', 'Left13.txt', 'Left14.txt', 'Left15.txt',
    'Right0.txt', 'Right1.txt', 'Right2.txt', 'Right3.txt', 'Right4.txt', 'Right5.txt', 'Right6.txt', 'Right7.txt', 
    'Right8.txt', 'Right9.txt', 'Right10.txt', 'Right11.txt', 'Right12.txt', 'Right13.txt', 'Right14.txt', 'Right15.txt',
    'Stay0.txt', 'Stay1.txt', 'Stay2.txt', 'Stay3.txt', 'Stay4.txt', 'Stay5.txt', 'Stay6.txt', 'Stay7.txt', 
    'Stay8.txt', 'Stay9.txt', 'Stay10.txt', 'Stay11.txt', 'Stay12.txt', 'Stay13.txt', 'Stay14.txt', 'Stay15.txt',
    'Up0.txt', 'Up1.txt', 'Up2.txt', 'Up3.txt', 'Up4.txt', 'Up5.txt', 'Up6.txt', 'Up7.txt', 
    'Up8.txt', 'Up9.txt', 'Up10.txt', 'Up11.txt', 'Up12.txt', 'Up13.txt', 'Up14.txt', 'Up15.txt',]

    header = 'channel1, channel2, channel3, channel4, channel5, channel6, channel7, channel8, channel9, channel10, channel11, channel12, channel13, channel14, channel15, channel15, direction\n'

    csv = open("dataCSV.csv", "w+")
    csv.write(header)

    for i in range(0, len(fileNames)):
        file = open("./TrainingData/TrainingDataWords/" + fileNames[i], "r")
        lines = file.readlines()    
        for line in lines:
            line = line.lstrip("[")
            line = line.strip("\n")
            line = line.rstrip("]")

            fname = os.path.basename(file.name).rstrip(".txt")
            fname = re.sub(r'\d+', '', fname)

            csv.write(line + "," + fname + "\n")



if __name__ == "__main__":
    main()