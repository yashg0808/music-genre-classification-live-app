import sys
import os
import dill
from src.exceptions import CustomException
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def model_evaluate(x_train, y_train, x_test, y_test, models):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(x_train, y_train)
            # y_test_pred = model.predict(x_test)
            r2_score = model.score(x_test, y_test)
            report[list(models.keys())[i]] = r2_score

        return report
            
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

