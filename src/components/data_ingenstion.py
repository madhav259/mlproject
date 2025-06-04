import os
import sys
import pandas as pd
from src.exception import customException
from src.logger import logging

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact','train.csv')
    test_data_path: str=os.path.join('artifact','test.csv')
    raw_data_path: str=os.path.join('artifact','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_Ingestion(self):
        logging.info('Enter the data ingestion component or method')
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the data as a data frame ')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('train_test_split initiated')
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=32)
            df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed')
            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        except Exception as e:
            raise customException(e,sys)

if __name__=='__main__':
    obj=DataIngestion()
    train_data,test_data = obj.initiate_data_Ingestion()
    data_transformation=DataTransformation()
    data_transformation.initiate_data_transformation(train_data,test_data)
    