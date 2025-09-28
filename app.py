from config import AppConfig
from modules import logger, exception
from summarizer.BertSummarizer import BartSummarizer



if __name__ == "__main__":
    summarizer = BartSummarizer(model_name=AppConfig.SUMM_MODELNAME, saved_model_dir=AppConfig.SUM_MODEL_DIR)
    summarizer.finetune(num_epochs=AppConfig.SUM_NUM_EPOCH)

