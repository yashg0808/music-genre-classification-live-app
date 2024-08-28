import os
import sys
from dataclasses import dataclass
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataPreprocessorConfig:
    preprocessor:str = os.path.join('artifacts/preprocessor.pkl')

class DataPreprocessor:
    def __init__(self):
        self.data_preprocessor_config = DataPreprocessorConfig()

    def preprocessing_steps(self):
        try:
            features = ['MFCCS2','MFCCS1','MFCCS3','MFCCS4','MFCCS5','MFCCS6','MFCCS7','MFCCS8','MFCCS9','MFCCS10','MFCCS11','MFCCS12','MFCCS13','MFCCS14','MFCCS15','MFCCS16','MFCCS17','MFCCS18','MFCCS19','MFCCS20']
            pipeline = Pipeline([('scaling', StandardScaler())])
            preprocessing_pipeline = ColumnTransformer([('preprocessor',pipeline,features)])
            return (preprocessing_pipeline)
        except Exception as e:
            raise CustomException(e,sys) 
    
    def preprocessing(self,train_data_path,test_data_path):
        logging.info('preprocessing initiated')
        try:
            preprocessor_obj = self.preprocessing_steps()
            train_data_mfccs = pd.read_csv(train_data_path)
            test_data_mfccs = pd.read_csv(test_data_path)
            arr = []
            for i in range(len(train_data_mfccs['Genre'])):
                if train_data_mfccs['Genre'][i]=='Funk':
                    arr.append(1)
                elif train_data_mfccs['Genre'][i]=='Rock':
                    arr.append(2)
                elif train_data_mfccs['Genre'][i]=='Hip Hop':
                    arr.append(3)
                elif train_data_mfccs['Genre'][i]=='Pop':
                    arr.append(4)
                elif train_data_mfccs['Genre'][i]=='Jazz':
                    arr.append(5)
                elif train_data_mfccs['Genre'][i]=='Romance':
                    arr.append(6)

            train_target_df = pd.DataFrame(arr, columns=['Genre'])

            arr = []
            for i in range(len(test_data_mfccs['Genre'])):
                if test_data_mfccs['Genre'][i]=='Funk':
                    arr.append(1)
                elif test_data_mfccs['Genre'][i]=='Rock':
                    arr.append(2)
                elif test_data_mfccs['Genre'][i]=='Hip Hop':
                    arr.append(3)
                elif test_data_mfccs['Genre'][i]=='Pop':
                    arr.append(4)
                elif test_data_mfccs['Genre'][i]=='Jazz':
                    arr.append(5)
                elif test_data_mfccs['Genre'][i]=='Romance':
                    arr.append(6)

            test_target_df = pd.DataFrame(arr, columns=['Genre'])
            
            train_data = train_data_mfccs.drop('Genre', axis = 1)
            test_data = test_data_mfccs.drop('Genre', axis = 1)

            train_data_scaled = preprocessor_obj.fit_transform(train_data)
            test_data_scaled = preprocessor_obj.transform(test_data)

            train_data_scaled_df = pd.DataFrame(train_data_scaled)
            test_data_scaled_df = pd.DataFrame(test_data_scaled)

            final_train_data = pd.concat([train_data_scaled_df, train_target_df], axis = 1)
            final_test_data = pd.concat([test_data_scaled_df, test_target_df], axis = 1)
            logging.info('preprocessing completed')

            save_object(
                file_path=self.data_preprocessor_config.preprocessor,
                obj = preprocessor_obj
            )
            return(final_train_data, final_test_data, self.data_preprocessor_config.preprocessor,)
        
        except Exception as e:
            raise CustomException(e,sys)



