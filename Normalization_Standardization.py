from sklearn.preprocessing import LabelEncoder, StandardScaler, Normalizer
import pandas as pd


class convertStr:
    """
    Convert string columns to numeric columns using:\n
    - Normalizer\n
    - StandardScaler\n
    - LabelEncoder\n
    The functions return the dataframe after modifications
    NOTE: Pass the dataframe without target column.\n
    """

    @staticmethod
    def normalize(dataframe: pd.DataFrame) -> pd.DataFrame:

        dataframe = convertStr().featureEncoder(dataframe)

        # normalize numeric features
        dataframe[dataframe.columns] = Normalizer().fit_transform(dataframe[dataframe.columns])

        return dataframe

    @staticmethod
    def standardize(dataframe: pd.DataFrame) -> pd.DataFrame:

        dataframe = convertStr().featureEncoder(dataframe)

        # normalize numeric features
        dataframe[dataframe.columns] = StandardScaler().fit_transform(dataframe[dataframe.columns])

        return dataframe

    @staticmethod
    def featureEncoder(dataframe: pd.DataFrame) -> pd.DataFrame:

        # get names of string columns
        stringColumns = []
        for col in dataframe.columns:
            if type(dataframe[col].values[0]) == str:
                stringColumns.append(col)

        # convert the columns that contain string into integers by transform labels to normalized encoding
        for c in stringColumns:
            lbl = LabelEncoder()
            lbl.fit(list(dataframe[c].values))
            dataframe[c] = lbl.transform(list(dataframe[c].values))

        return dataframe