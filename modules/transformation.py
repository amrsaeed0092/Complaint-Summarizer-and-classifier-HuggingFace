from modules.ingestion import DataIngestion
from config import AppConfig


class DataTransformer:
    def __init__(self):
        #downloading the dataset
        downloader =DataIngestion(dataset_name=AppConfig.SUMM_DATASETNAME) 
        self.train_df, self.val_df, self.test_df = downloader.load_data()
        
        
    def preprocessed(self):
        self.train_df = self.train_df.drop(['text'], axis=1)
        self.test_df = self.test_df.drop(['text'], axis=1) 
        self.val_df = self.val_df.drop(['text'], axis=1)   
        return self.train_df, self.val_df, self.test_df