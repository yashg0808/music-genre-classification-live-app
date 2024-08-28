import os
import sys
from dataclasses import dataclass
from src.logger import logging
from src.exceptions import CustomException

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.components.data_preprocessing import DataPreprocessorConfig
from src.components.data_preprocessing import DataPreprocessor

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join('artifacts/data.csv')
    train_data_path:str = os.path.join('artifacts/train.csv')
    test_data_path:str = os.path.join('artifacts/test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def data_ingestion(self):
        logging.info('data ingestion initiated')
        try:
            data = pd.read_csv(r'notebooks\data\MFCCS.csv')
            logging.info('dataset collected')
            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path)
            train_set, test_set = train_test_split(data, random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path)
            test_set.to_csv(self.data_ingestion_config.test_data_path)
            logging.info('train and test data splitted')

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    data_ingest = DataIngestion()
    train_data, test_data = data_ingest.data_ingestion()

    data_preprocess = DataPreprocessor()
    final_train_data, final_test_data, _ = data_preprocess.preprocessing(train_data, test_data)
    
    model_train = ModelTrainer()
    print(model_train.model_training(final_train_data, final_test_data))
