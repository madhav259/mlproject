import sys
import os
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.logger import logging
from src.exception import customException
import numpy as np
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformations
        '''
        try:
            numerical_features=['reading_score','writing_score']
            categorical_features=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
                ]
            num_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(strategy='median')),
                       ('scalar',StandardScaler())]
            )
            cat_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(strategy='most_frequent')),(
                    'encoder',OneHotEncoder()),
                    ('scaling',StandardScaler(with_mean=False))]
            )
            logging.info('categorical columns standard scaling computed')
            logging.info('numerical Standard scaling completed')

            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_features),('cat_pipeline',cat_pipeline,categorical_features)
            ])
            return preprocessor
        except Exception as e:
            raise customException(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info('read train and test data completed')
            logging.info('Obtaining preprocessing  object')

            preprocessing_object=self.get_data_transformer_object()

            target_column_name='math_score'
            numerical_columns=['reading_score','writing_score']

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info('applying preprocession on my training and testing dataset ')
            input_feature_train_arr=preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_object.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info('saved preprocesing object ')
            save_obj(file_path=self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_object)
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            raise customException(e,sys)