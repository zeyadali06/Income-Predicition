import pandas as pd


class dataCorrection:


    @staticmethod
    def clean(dataframe: pd.DataFrame) -> pd.DataFrame:
        
        dataframe = dataCorrection()._removeGarbage(dataframe)

        dataframe = dataCorrection().__replaceWrongData(dataframe)

        dataframe['Income'] = dataframe['Income'].replace(['<=50K', '>50K', '<=50K.', '>50K.'], [0, 1, 0, 1])

        return dataframe

    @staticmethod
    def _removeGarbage(dataframe: pd.DataFrame) -> pd.DataFrame:
        dataframe = dataframe.dropna()
        dataframe = dataframe.drop_duplicates()
        return dataframe

    @staticmethod
    def __replaceWrongData(dataframe: pd.DataFrame) -> pd.DataFrame:
        for col in dataframe.columns:
            try:
                dataframe[col] = pd.to_numeric(dataframe[col])
            except ValueError:
                dataframe[col] = dataframe[col].str.replace('?', f"{str.split(str(dataframe[col].mode()))[1]}")
        return dataframe
