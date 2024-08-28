import os
import sys
from dataclasses import dataclass
from collections import Counter

from src.exceptions import CustomException
from src.logger import logging
from src.utils import model_evaluate, save_object

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

@dataclass
class ModelTrainerConfig:
    model:str = os.path.join('artifacts/model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def model_training(self, final_train_data, final_test_data):
        logging.info('model training initiated')
        try:
            x_train, y_train, x_test, y_test = (
                final_train_data.iloc[:,:20],
                final_train_data.iloc[:,20:],
                final_test_data.iloc[:,:20],
                final_test_data.iloc[:,20:]
            )
            

            models = {
                'SVM' : SVC(),
                'Decision Tree' : DecisionTreeClassifier(),
                'Random Forest' : RandomForestClassifier()
            }

            model_report:dict = model_evaluate(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, models=models)
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.model,
                obj = best_model
            )
            y_pred = best_model.predict(x_test)
            logging.info('model training completed')
            # return classification_report(y_test, y_pred)
            most_common_element = Counter(y_pred).most_common(1)[0][0]
            # if most_common_element == 1:
            #     return ('Funk')
            # elif most_common_element == 2:
            #     return ('Rock')
            # elif most_common_element == 3:
            #     return ('Hip Hop')
            # elif most_common_element == 4:
            #     return ('Pop')
            # elif most_common_element == 5:
            #     return ('Jazz')
            # elif most_common_element == 6:
            #     return ('Romance')

            return most_common_element

        
        except Exception as e:
            raise CustomException(e,sys)