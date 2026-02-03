import sys
import os
import pandas as pd

from src.exception import CustomException
from src.utlis import load_object

class predictPipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:
            model_path=os.path.join('artifacts',"model.pkl")
            preprocessor_path=os.path.join('artifacts',"preprocessor.pkl")

            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)

            data_scaled=preprocessor.transform(feature)
            preds=model.predict(data_scaled)

            return preds
        

        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:
    def __init(self,gender:str):
        self.gender

    def get_data_as_data_frame(self):
        try:
                custom_data_input_dict = {
                "gender": [self.gender],
                
            }
                return pd.DataFrame(custom_data_input_dict)
                

                     

        except Exception as e:
            raise CustomException(e,sys)
                
               
