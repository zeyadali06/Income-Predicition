from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from typing import Literal
import numpy as np
import pandas as pd


class training:
    

    @staticmethod
    def trainPredict(xtrain: pd.DataFrame, ytrain: pd.DataFrame, xtest: pd.DataFrame, way: Literal['lg', 'svc', 'dtc', 'rfc'] | Literal[1, 2, 3, 4]) -> np.ndarray:
        """
        Fit data to the classification model using one of the next models:\n
        1- Logistic Regression.\n
        2- Support Vector Classifier.\n
        3- Decision Tree Classifier.\n
        4- Random Forest Classifier.\n
        Parameters
        ----------
        - xtrain: The features of the train dataframe without the target column.\n
        - ytrain: The target column of the train dataframe.\n
        - xtest: The features of the test dataframe without the target column.\n
        - way: The way of the model ['lg', 'svc', 'dtc', 'rfc'] or [1, 2, 3, 4].\n
        Return
        ------
        np.ndarray\n
        An array of the Y predication.
        """
        
        # to set datatype of way as int if possible
        try:
            way = int(way) 
        except ValueError:
            way = way.casefold()  

        gridResult = training.__grid(xtrain, ytrain, way)

        # determine the model
        if way == 'lg' or way == 1:
            model = LogisticRegression(solver=gridResult['solver'], penalty=gridResult['penalty'], C=gridResult['C'])
        elif way == 'svc' or way == 2:
            model = SVC(kernel=gridResult['kernel'], max_iter=gridResult['max_iter'])
        elif way == 'dtc' or way == 3:
            model = DecisionTreeClassifier(max_leaf_nodes=gridResult['max_leaf_nodes'], criterion=gridResult['criterion'])
        elif way == 'rfc' or way == 4:
            model = RandomForestClassifier(max_depth=gridResult['max_depth'], criterion=gridResult['criterion'], n_estimators=gridResult['n_estimators'])

        model.fit(xtrain, ytrain) 
        YPred = model.predict(xtest) 

        return YPred

    @staticmethod
    def __grid(xtrain: pd.DataFrame, ytrain: pd.DataFrame, way: Literal['lg', 'svc', 'dtc', 'rfc'] | Literal[1, 2, 3, 4]):

        # define models and hyperparameters
        if way == 'lg' or way == 1:
            model = LogisticRegression()
            param_dict = dict(solver=['lbfgs', 'liblinear'], penalty=['l1', 'l2'], C=[100, 10 , 1, 0.1])

        elif way == 'svc' or way == 2:
            model = SVC()
            param_dict = dict(kernel = ['poly', 'rbf'], max_iter = [5000, 8000, -1])

        elif way == 'dtc' or way == 3:
            model = DecisionTreeClassifier()
            param_dict = dict(max_leaf_nodes = np.arange(10,30), criterion = ['gini', 'entropy', 'log_loss'])

        elif way == 'rfc' or way == 4:
            model = RandomForestClassifier()
            param_dict = dict(n_estimators= np.arange(10, 100, 8), max_depth = np.arange(5, 20, 3), criterion = ['gini', 'entropy', 'log_loss'])

        # define grid search and get best hyperparameter
        grid_result = GridSearchCV(estimator=model, param_grid=param_dict, cv=4, verbose=3).fit(xtrain, ytrain) 

        return grid_result.best_estimator_.get_params()


