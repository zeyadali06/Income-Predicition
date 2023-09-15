import numpy as np
import pandas as pd


class outliers:
    
    @staticmethod
    def replace(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Replace outliers in numerical columns with the mode of the column.\n
        Parameters
        ----------
        - dataframe: Dataframe you want to replace outliers in it.\n
        Return
        ------
        pd.DataFrame\n
        The dataframe after replace outliers.
        """

        cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

        # replace outlires in each column
        for c in cols:
            mode = float(str.split(str(dataframe[c].mode()))[1])
            lower, upper = outliers().__lowerUpper(dataframe[c])
            dataframe[c] = dataframe[c].apply(lambda x: mode if (x < lower or x > upper) else x)
            
        return dataframe

    @staticmethod
    def remove(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that contain outliers in numerical columns.\n
        Parameters
        ----------
        - dataframe: Dataframe you want to remove outliers from it.\n
        Return
        ------
        dataframe\n
        The dataframe after deleting rows that contain outliers.
        """

        cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']

        # drop rows that contain outliers
        for c in cols:  
            remove = outliers().__detectOutliers(dataframe[c])
            for i in remove.reshape(1, len(remove)):
                dataframe = dataframe.drop(i)

        return dataframe

    @staticmethod
    def __detectOutliers(series: pd.Series) -> np.ndarray:
        lower, upper = outliers().__lowerUpper(series)

        # indexs of rows that contain outliers 
        upperArray = np.array(series.index[series > upper]) 
        lowerArray = np.array(series.index[series < lower])

        return np.concatenate((upperArray, lowerArray))

    @staticmethod
    def __lowerUpper(series: pd.Series) -> tuple[float, float]:
        q1 = series.quantile(.25)
        q3 = series.quantile(.75)
        IQR = q3 - q1
        upper = q3 + 1.5 * IQR  # upper bound
        lower = q1 - 1.5 * IQR  # lower bound

        return lower, upper
