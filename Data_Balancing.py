from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from typing import Literal
import pandas as pd


class Balancing:

    @staticmethod
    def resample(dataframe, target:str, way:Literal['under', 'over']):
        """
        Balancing values of target column.\n
        NOTE: The number of rows migth be deferent.\n
        Parameter
        ---------
        - dataframe: Dataframe you want to resample it.\n
        - target: the name of the target column.\n
        - way: The way of the resampling
        Return
        ------
        pd.DataFrame\n
        The dataframe after modifications.
        """
        
        # specify the model
        if way == 'under':
            model = RandomUnderSampler(random_state=0, replacement=True)
        elif way == 'over':
            model = RandomOverSampler(random_state=0)
        
        #resampling data
        x_rus, y_rus = model.fit_resample(dataframe[dataframe.columns[:-1]], dataframe[target])
        
        # make a dataframe of the data after resampling
        df = pd.DataFrame()
        for c in dataframe.columns[:-1]:
            df[c] = x_rus[c]        
        df[target] = y_rus
        
        return df
            
    