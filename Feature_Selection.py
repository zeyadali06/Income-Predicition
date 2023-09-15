import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, RFE, f_classif
from typing import Literal


class featureSelection:

    @staticmethod
    def technique(trainData: pd.DataFrame,
                  testData: pd.DataFrame,
                  name: Literal['us', 'fi', 'rfe', 'iwc'] | Literal[1, 2, 3, 4] = 1,
                  k: int = 10,
                  another: Literal['commenFeatures', 'suggestion'] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function enables you to choose a technique from the next techniques:\n
        1- Univariate Selection(US).\n
        2- Feature Importance(FI).\n
        3- Recursive Feature Elimination(RFE).\n
        4- Ignore Weak Correlation(IWC).\n
        then drop the features from the dataframe that affecting in an unnoticed way without changing values of target column.\n
        NOTE: Pass the dataframe with the target column.\n
        Parameters
        ----------
        - trainData: The train dataframe you want to select features from.\n
        - testData: The test dataframe(to remove columns removed from train dataframe).\n
        - name: Technique's name or string of the number ['us', 'fi', 'rfe', 'iwc'] or [1, 2, 3, 4].\n
            1- us: Univariate Selection.\n
            2- fi: Feature Importance.\n
            3- rfe: Recursive Feature Elimination.\n
            4- iwc: Ignore Weak Correlation.\n
        - k: Number of features you want to be in the dataframe (between 0 and 15)(default = 10).\n
        - another: = commenFeatures: The common selected features from all of the techniques(default = None).\n
                   = suggestion:Remove the following rows from the dataframe['fnlwgt', 'education', 'relationship', 'race'].\n
        Return
        ------
        train DataFrame, test DataFrame\n
        The train dataframe and test dataframe after modifications.
        """

        if another == 'suggestion':
            try:
                trainData = trainData.drop(['fnlwgt', 'education', 'relationship', 'race'], axis=1)
                testData = testData.drop(['fnlwgt', 'education', 'relationship', 'race'], axis=1)
            except KeyError:
                trainData = trainData.drop(['fnlwgt'], axis=1)
                testData = testData.drop(['fnlwgt'], axis=1)
            return trainData, testData
        
        selectedFeatures = []
        traintargetCol = trainData['Income']
        testtargetCol = testData['Income']

        # to set datatype of name as int if possible
        try:
            name = int(name) 
        except ValueError:
            name = name.casefold()

        # check validation of some parameters
        if name not in ['us', 'fi', 'rfe', 'iwc', 1, 2, 3, 4]:
            raise ValueError("Invalid value.\n\tname must equil one of these ['us', 'fi', 'rfe', 'iwc', 1, 2, 3, 4]")
        if k <= 0 or k >= 15:
            raise ValueError("Invalid value.\n\tk must be between 0 and 15")

        if another == 'commenFeatures':
            all = []  # to set all features from all techniques in it
            us = featureSelection().__UnivariateSelection(trainData, k) 
            fi = featureSelection().__FeatureImportance(trainData, k)
            lr = featureSelection().__IgnoreWeakCorrelation(trainData, k)
            iwc = featureSelection().__RecursiveFeatureElimination(trainData, k)
            all.extend(us)
            all.extend(fi)
            all.extend(lr)
            all.extend(iwc)
            # add common features to selectedFeatures list
            for f in all:
                if f in us and f in fi and f in lr and f in iwc:
                    selectedFeatures.append(f)
        elif name == 'us' or name == 1:
            selectedFeatures.extend(featureSelection().__UnivariateSelection(trainData, k)) 
        elif name == 'fi' or name == 2:
            selectedFeatures.extend(featureSelection().__FeatureImportance(trainData, k))
        elif name == 'rfe' or name == 3:
            selectedFeatures.extend(featureSelection().__RecursiveFeatureElimination(trainData, k))
        elif name == 'iwc' or name == 4:
            selectedFeatures.extend(featureSelection().__IgnoreWeakCorrelation(trainData, k))

        # drop unchoosen features from train and test dataframes
        for f in trainData.columns:
            if f not in selectedFeatures:
                trainData = trainData.drop([f], axis=1)
                testData = testData.drop([f], axis=1)

        # add target column to both dataframes
        trainData['Income'] = traintargetCol
        testData['Income'] = testtargetCol

        return trainData, testData

    @staticmethod
    def __UnivariateSelection(dataframe: pd.DataFrame, num: int) -> list:
        NoColumns = len(dataframe.columns)
        X = dataframe.iloc[:, 0:NoColumns-1]  # features columns
        Y = dataframe.iloc[:, NoColumns-1]  # target column

        # select featureas by using f_classif method
        selectedFeatures = X.columns[SelectKBest(score_func=f_classif, k=int(num)).fit(X, Y).get_support()]

        return list(selectedFeatures)

    @staticmethod
    def __RecursiveFeatureElimination(dataframe: pd.DataFrame, k: int) -> list:
        NoColumns = len(dataframe.columns)
        X = dataframe.iloc[:, 0:NoColumns-1]  # features columns
        Y = dataframe.iloc[:, NoColumns-1]  # target column

        # select featureas by using recursive feature elimination(RFE) class
        rfe = RFE(LogisticRegression(), n_features_to_select=k)
        fit = rfe.fit(X, Y)
        selectedFeatures = []

        # add features to selectedFeatures list
        for key, index in enumerate(fit.support_):
            if index == True:
                selectedFeatures.append(dataframe.columns[key])

        return selectedFeatures

    @staticmethod
    def __FeatureImportance(dataframe: pd.DataFrame, k: int) -> list:
        NoColumns = len(dataframe.columns)
        X = dataframe.iloc[:, 0:NoColumns-1]  # features columns
        Y = dataframe.iloc[:, NoColumns-1]  # target column
        featuresImportance = pd.Series(ExtraTreesClassifier().fit(X, Y).feature_importances_)

        # create dataframe with two columns, first column contain names of features and another contain importance values
        df = pd.DataFrame()
        df['Features'] = dataframe.columns[:NoColumns-1]
        df['Importance'] = featuresImportance

        # sort the importance column to extract the largest k features
        df = df.sort_values(axis=0, ascending=False, by=['Importance'])
        selectedFeatures = df['Features'].iloc[:k]

        return list(selectedFeatures)

    @staticmethod
    def __IgnoreWeakCorrelation(dataframe: pd.DataFrame, k: int) -> list:
        NoColumns = len(dataframe.columns)
        corrMatrix = dataframe.corr().abs()

        # create dataframe with two columns, first column contain names of features and another contain correlation values
        df = pd.DataFrame()
        df['Features'] = corrMatrix.columns[:NoColumns-1]
        df['Correlation'] = corrMatrix['Income'].iloc[:NoColumns-1].values

        # sort the importance column to extract the largest k features
        df = df.sort_values(axis=0, ascending=False, by=['Correlation'])
        selectedFeatures = df['Features'].iloc[:k]

        return list(selectedFeatures)
