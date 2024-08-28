import os
import sys
import dill
from src.exceptions import CustomException
from src.logger import logging
from src.utils import load_object
from dataclasses import dataclass
import pandas as pd
class ModelTrainPipeline:
    def __init__(self):
        pass
    def predict_pipeline(self,features):
        try:
            model = load_object(file_path = 'artifacts/model.pkl')
            preprocessor = load_object(file_path = 'artifacts/preprocessor.pkl')
            preprocessed_features = preprocessor.transform(features)
            preds = model.predict(preprocessed_features)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        

class PredictData:
    def __init__(self, MFCCS2:float,MFCCS1:float,MFCCS3:float,MFCCS4:float,MFCCS5:float,MFCCS6:float,MFCCS7:float,MFCCS8:float,MFCCS9:float,MFCCS10:float,MFCCS11:float,MFCCS12:float,MFCCS13:float,MFCCS14:float,MFCCS15:float,MFCCS16:float,MFCCS17:float,MFCCS18:float,MFCCS19:float,MFCCS20:float):
        self.MFCCS2 = MFCCS2
        self.MFCCS1 = MFCCS1
        self.MFCCS3 = MFCCS3
        self.MFCCS4 = MFCCS4
        self.MFCCS5 = MFCCS5
        self.MFCCS6 = MFCCS6
        self.MFCCS7 = MFCCS7
        self.MFCCS8 = MFCCS8
        self.MFCCS9 = MFCCS9
        self.MFCCS10 = MFCCS10
        self.MFCCS11 = MFCCS11
        self.MFCCS12 = MFCCS12
        self.MFCCS13 = MFCCS13
        self.MFCCS14 = MFCCS14
        self.MFCCS15 = MFCCS15
        self.MFCCS16 = MFCCS16
        self.MFCCS17 = MFCCS17
        self.MFCCS18 = MFCCS18
        self.MFCCS19 = MFCCS19
        self.MFCCS20 = MFCCS20

    def get_data_as_df(self):
        try:
            data_dict = {
                'MFCCS2' : [self.MFCCS2],
                'MFCCS1' : [self.MFCCS1],
                'MFCCS3' : [self.MFCCS3],
                'MFCCS4' : [self.MFCCS4],
                'MFCCS5' : [self.MFCCS5],
                'MFCCS6' : [self.MFCCS6],
                'MFCCS7' : [self.MFCCS7],
                'MFCCS8' : [self.MFCCS8],
                'MFCCS9' : [self.MFCCS9],
                'MFCCS10' : [self.MFCCS10],
                'MFCCS11' : [self.MFCCS11],
                'MFCCS12' : [self.MFCCS12],
                'MFCCS13' : [self.MFCCS13],
                'MFCCS14' : [self.MFCCS14],
                'MFCCS15' : [self.MFCCS15],
                'MFCCS16' : [self.MFCCS16],
                'MFCCS17' : [self.MFCCS17],
                'MFCCS18' : [self.MFCCS18],
                'MFCCS19' : [self.MFCCS19],
                'MFCCS20' : [self.MFCCS20]

            }
            data_df = pd.DataFrame(data_dict)
            return data_df
        except Exception as e:
            raise CustomException(e,sys)
        

