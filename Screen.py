from Data_Balancing import *
from Normalization_Standardization import *
from Training import *
from Feature_Selection import *
from Remove_Outliers import *
from Clean_data import *
from Visualize import *
from sklearn import metrics
import datetime
import pandas as pd


class screen:

    def __init__(self) -> None:
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.ypred = np.ndarray([])
        self.allInOne = {'outliers': str,
                         'convertToNum': str,
                         'featureSelection': {'way': int, 'NoFeatures': 0, 'technique': int, 'features': list}, 
                         'balancing': str,
                         'training': int,
                         'evaluations': {'report': (str | dict), 'error': float, 'confusion': np.ndarray, 'time': 0}}
    
    
    def dataInfo(self) -> None:
        print("Do you want to show some statistics about data?(y/n)")
        while True:
            c = input().casefold().strip()
            if c == 'y':
                print("Information about trainig data")
                print(self.train.info())
                print("Information about testing data")
                print(self.test.info())
                break
            elif c == 'n':
                break
            else:
                print('Wrong answer. Please reanswer the question: ', end='')
                continue
            

    def outliersBoxplot(self) -> None:
        print("Do you want to show Boxplots for outliers of numeric columns before gitting rid of outliers?(y/n)")
        while True:
            c = input().casefold().strip()
            if c == 'y':
                visualize().showOutliers(self.train[self.train.columns[:-1]])
                break
            elif c == 'n':
                break
            else:
                print('Wrong answer. Please reanswer the question: ', end='')
                continue
            
            
    def gitRideOfOutliers(self) -> None:
        print("Do you want to replace(p) or remove(m) outliers(replace is recommeneded)?(p/m)")
        while True:
            c = input().casefold().strip()
            if c == 'p':
                self.train = outliers().replace(self.train)
                self.test = outliers().replace(self.test)
                print('Outliers has been replaced.')
            elif c == 'm':
                self.train = outliers().remove(self.train)
                self.test = outliers().remove(self.test)
                print('Outliers has been removed.')
            else:
                print('Wrong answer. Please reanswer the question: ', end='')
                continue
            self.allInOne['outliers'] = c
            break
            
            
    def corrHeatmap(self) -> None:
        print("Do you want to show Heatmap for correlation between numeric features?(y/n)")
        # print(self.train['workclass'].iloc[0])
        while True:
            c = input().casefold().strip()
            if c == 'y':
                visualize().showCorrHeatmap(self.train)
                break
            elif c == 'n':
                break
            else:
                print('Wrong answer. Please reanswer the question: ', end='')
                continue
            
            
    def standOrNorm(self) -> None:
        print("Do you want to stanardize(s) or normalize(n) data(standardize is recommended)?(s/n)")
        while True:
            c = input().casefold().strip()
            if c == 's':
                self.train[self.train.columns[:-1]] = convertStr().standardize(self.train[self.train.columns[:-1]])
                self.test[self.test.columns[:-1]] = convertStr().standardize(self.test[self.test.columns[:-1]])
            elif c == 'n':
                self.train[self.train.columns[:-1]] = convertStr().normalize(self.train[self.train.columns[:-1]])
                self.test[self.test.columns[:-1]] = convertStr().normalize(self.test[self.test.columns[:-1]])
            else:
                print('Wrong answer. Please reanswer the question: ', end='')
                continue
            self.allInOne['convertToNum'] = c
            print('Data has been stanardized')
            break
            
            
    def featureSelection(self) -> None:
        print('There are some feature selection techniques:')
        print('\t1- Univariate Selection.\n\t2- Feature Importance.\n\t3- Recursive Feature Elimination.\n\t4- Ignore Weak Correlation.\n')
        print("Select away from the next, then enter it's number:", end='')
        print("\n\t1- Select common featuers from four technique.\n\t2- Make the program suggest the features.\n\t3- Choose a technique from the above.")
        print("Note: In the first two ways, the number of features may be less than the number of features you want.")
        while True:
            c = int(input().strip())
            if c == 1:
                print("Enter number of features to be selected(from 1 to 14): ",end='') 
                while True:
                    num = int(input().strip())
                    if num > 14 or num < 1:
                        print('Wrong answer. Please reanswer the question: ',end='')
                        continue
                    else:
                        break
                print('Please wait...')
                self.train, self.test = featureSelection().technique(trainData=self.train, testData=self.test, another='commenFeatures', k=num)
                
            elif c == 2:
                self.train, self.test = featureSelection().technique(trainData=self.train, testData=self.test, another='suggestion')
                
            elif c == 3:
                print("Enter number of features to be selected(from 1 to 14): ",end='') 
                while True:
                    num = int(input().strip())
                    if num > 14 or num < 1:
                        print('Wrong answer. Please reanswer the question: ', end='')
                        continue
                    else:
                        self.allInOne['featureSelection']['NoFeatures'] = num
                        break
                    
                print("Enter technique's number: ", end='')
                while True:
                    technique = int(input().strip())
                    if technique == 1 or technique == 2 or technique == 3 or technique == 4:
                        print('Please wait...')
                        self.train, self.test = featureSelection().technique(trainData=self.train, testData=self.test, name=technique, k=num)  
                        self.allInOne['featureSelection']['technique'] = technique
                        break
                    else:
                        print('Wrong answer. Please reanswer the question: ', end='')
                        continue
                
            else:
                print('Wrong answer. Please reanswer the question: ', end='')
                continue
            
            self.allInOne['featureSelection']['way'] = c
            self.allInOne['featureSelection']['features'] = list(self.train.columns[:-1])
            print(f'Features has been selected: {list(self.train.columns[:-1])}')
            break
        
               
    def dataBalancing(self) -> None:
        # Printing data before balancing
        print("Training data target column statistics:")
        print(self.train['Income'].value_counts())
        print("Testing data target column statistics:")
        print(self.test['Income'].value_counts())

        # Balancing data
        print("Do you want to balancing data using Under(u) or Over(o) resampling?(u/o)")
        while True:
            c = input().strip().casefold()
            if c == 'u':
                self.train = Balancing().resample(self.train, 'Income', 'under')
                self.test = Balancing().resample(self.test, 'Income', 'under')
                
            elif c == 'o':
                self.train = Balancing().resample(self.train, 'Income', 'over')
                self.test = Balancing().resample(self.test, 'Income', 'over')
                
            else:
                print('Wrong answer. Please reanswer the question: ', end='')
                continue
            
            self.allInOne['balancing'] = c
            break
            
        # Printing data after balancing
        print("Training data target column statistics:")
        print(self.train['Income'].value_counts())
        print("Testing data target column statistics:")
        print(self.test['Income'].value_counts())


    def training(self) -> None:
        X_train = self.train[self.train.columns[:-1]]
        Y_train = self.train[['Income']]
        xTest = self.test[self.test.columns[:-1]]
        print('There are some ways for training data:')
        print('\t1- Logistic Regression\n\t2- Support Vector Classifier\n\t3- Decision Tree Classifier\n\t4- Random Forest Classifier')

        while True:
            c = int(input().strip())
            if c == 1 or c == 2 or c == 3 or c == 4:
                self.allInOne['training'] = c
                print('Please wait, this will take afew minutes...')
                startTime = datetime.datetime.now()
                self.ypred = training().trainPredict(X_train, Y_train, xTest, c)
                self.allInOne['evaluations']['time'] = datetime.datetime.now() - startTime
                
            else:
                print('Wrong answer. Please reanswer the question: ', end='')
                continue
            self.allInOne['training'] = c
            break
           
                
    def evaluation(self) -> None:
        print("Do you want to show some evaluation about model?(y/n)")
        self.yTest = self.test[['Income']]
        self.allInOne['evaluations']['report'] = metrics.classification_report(self.yTest, self.ypred)
        self.allInOne['evaluations']['error'] = metrics.mean_squared_error(self.yTest, self.ypred)
        self.allInOne['evaluations']['confusion'] = metrics.confusion_matrix(self.yTest, self.ypred)
        while True:
            c = input().strip().casefold()
            if c == 'y':
                print(self.allInOne['evaluations']['report'])
                print('Time of Training Model: ', self.allInOne['evaluations']['time'])
                print('Mean Squared Error: ', self.allInOne['evaluations']['error'])
                print('Confusion Matrix: \n', self.allInOne['evaluations']['confusion'])
                break
            elif c == 'n':
                break
            else:
                print('Wrong answer. Please reanswer the question: ', end='')
                continue
            
                 
    def screen(self ,train:pd.DataFrame, test:pd.DataFrame) -> None:
        self.train = train
        self.test = test
        print('\n\t\t\t\t*****    Some Information about data    *****\n')
        screen.dataInfo(self)
        
        print('\n\t\t\t\t\t*****    Outliers    *****\n')
        screen.outliersBoxplot(self)
        screen.gitRideOfOutliers(self)
        
        print('\n\t\t\t\t\t*****    Correlation    *****\n')
        screen.corrHeatmap(self)
        
        print('\n\t\t\t\t*****    Converting Data to Numeric    *****\n')
        screen.standOrNorm(self)
        
        print('\n\t\t\t\t*****    Feature Selection    *****\n')
        screen.featureSelection(self)
        
        print('\n\t\t\t\t\t*****    Data Balancing    *****\n')
        screen.dataBalancing(self)
        
        print('\n\t\t\t\t\t*****    Training    *****\n')
        screen.training(self)
        
        print('\n\t\t\t\t*****    Some Evaluation of Model    *****\n')
        screen.evaluation(self)
        
    @staticmethod
    def showAll(allData: list[dict]) -> None:
        for key, data in enumerate(allData):
            outliers = 'replaced' if(data['outliers'] == 'p') else 'removed'
            convertToNum = 'standardize' if (data['convertToNum'] == 's') else 'normalize'
            way = 'select common features' if (data['featureSelection']['way'] == 1) else 'suggest features' if (data['featureSelection']['way'] == 2) else 'choose a technique'
            NoFeatures = data['featureSelection']['NoFeatures']
            technique = 'univariate selection' if (data['featureSelection']['technique'] == 1) else 'feature importance' if (data['featureSelection']['technique'] == 2) else 'recursive feature elimination' if (data['featureSelection']['technique'] == 3) else 'ignore weak correlation' if (data['featureSelection']['technique'] == 4) else ''
            features = data['featureSelection']['features']
            balance =  'under resampling' if (data['balancing'] == 'u') else 'over resampling'
            training = 'logistic regression' if (data['training'] == 1) else 'support vector classifier' if (data['training'] == 2) else 'decision tree classifier' if (data['training'] == 3) else 'random forest classifier'
            report = data['evaluations']['report']
            error = data['evaluations']['error']
            confusion = data['evaluations']['confusion']
            time = data['evaluations']['time']
            
            print('\t\t\t*******************\t', end='')
            print('1st model' if(key == 0) else '2nd model' if(key == 1) else f'{key + 1}th model', end='')
            print('\t*******************')
            
            print(f'Outliers: {outliers.capitalize()}\nStandardization or Normalization: {convertToNum.capitalize()}')
            print(f'Way of selecting features: {way.capitalize()}')
            if technique != '':
                print(f'Technique of selecting features: {technique.capitalize()}')
            print(f'Features: {features} = {NoFeatures}')
            print(f'Data Balancing: {balance.capitalize()}')
            print(f'Training model: {training.capitalize()}')
            print(report)
            print(f'Confusion Matrix: \n{confusion}')
            print(f'Error: {error}')
            print(f'Time of training: {int(time.total_seconds()/60)}:', end='')
            print(time.total_seconds()%60)
            
            
            