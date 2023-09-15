import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class visualize:

    @staticmethod
    def showCorrHeatmap(dataframe:pd.DataFrame) -> None:
    
        
        numericCols = visualize().__getNumCols(dataframe)
        plt.figure(figsize=(10, 7))
        sns.set_style('whitegrid')
        sns.heatmap(dataframe[numericCols].corr().abs().dropna(axis=0, how='all').dropna(axis=1, how='all'), annot=True)
        plt.show()

    @staticmethod
    def showOutliers(data: pd.DataFrame | pd.Series):
     
        
        numericCols = visualize().__getNumCols(data)
        sns.set_style('whitegrid')
        if len(numericCols) <= 7:
            
            plt.figure(figsize=(len(numericCols)*2, 4))
            plt.subplots_adjust(wspace=.6, left=.08, right=.92)
            for i, c in enumerate(numericCols):
                plt.subplot(1, len(numericCols), i+1)
                plt.title(c)
                
                try:
                    sns.boxplot(data[c])
                except KeyError:
                    sns.boxplot(data)
        else:
            plt.figure(figsize=(len(numericCols), 6.5))
            plt.subplots_adjust(wspace=.6, left=.07,
                                right=.93, bottom=.05, top=.95)
            for i, c in enumerate(numericCols):
                plt.subplot(2, 7, i+1)
                plt.title(c)
                try:
                    sns.boxplot(data[c])
                except KeyError:
                    sns.boxplot(data)

        plt.show()

        
    @staticmethod
    def __getNumCols(data:pd.DataFrame | pd.Series) -> list:
        numericCols = []
        try:
            for col in data.columns:
                if type(data[col].iloc[0]) in [np.float64, np.float32, np.int64, np.int32]:
                    numericCols.append(col)
        except KeyError | AttributeError :
            if type(data.iloc[0]) in [np.float64, np.float32, np.int64, np.int32]:
                numericCols.append(data.name)
        
        return numericCols
        
