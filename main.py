from Clean_data import *
from Screen import *
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=Warning)

mainDataTrain = pd.read_csv("trainData.csv", skipinitialspace=True)
mainDataTest = pd.read_csv("testData.csv", skipinitialspace=True)

train = dataCorrection().clean(mainDataTrain)
test = dataCorrection().clean(mainDataTest)

allmodels = []

print("\t\t\t\t\t\t\t\t*****  Welcome to Income Prediction Program  *****\n")
while True:
    model1 = screen()
    model1.screen(train, test)
    allmodels.append(model1.allInOne)
    print('Do you want to train another model?(y/n)')
    c = input().strip().casefold()
    if c == 'y':
        continue
    elif c == 'n':
        print('Thanks for your time')
        break
    else:
        print('Wrong answer. Please reanswer the question: ', end='')
        continue

screen().showAll(allmodels)

