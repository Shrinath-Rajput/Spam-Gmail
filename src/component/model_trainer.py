import os
import sys

import pandas as pd
df=pd.read_csv('archivedata/emails.csv')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix ,classification_report,r2_score




from src.exception import CustomException
from src.logger import logging
from src.utlis import save_object ,evaluate_models 

class ModelTrainerConfig:
    trained_model=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split the data of the dataset")
            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],#x
                train_array[:,-1],#y
                test_array[:,:-1],
                test_array[:,-1]
         )
            
            logi=LogisticRegression()
            logi.fit(x_train,y_train)



            #predict model
            y_pred=logi.predict(y_test)
            print(y_pred)

            #probability
            pro=logi.predict_log_proba(x_test)
            print(pro)


            #performance matrix
            acc_sco=accuracy_score(y_test,y_pred)
            print(acc_sco)

            con_matr=confusion_matrix(y_pred,y_test)
            print(con_matr)

            claasi=classification_report(y_pred,y_test)
            print(claasi)

            r2=r2_score(y_pred,y_test)
            print(r2)


            save_object(
                file_path=self.model_trainer_config.trained_model,
            )
               
            

        







        except Exception as e:
            raise CustomException(e,sys)


 
