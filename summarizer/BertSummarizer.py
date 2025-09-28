from modules.transformation import DataTransformer
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, pipeline
import torch
import os, sys
from modules import exception, logger
from datasets import DatasetDict, Dataset
from config import AppConfig
from time import time
import evaluate
import numpy as np

class BartSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn", saved_model_dir="./finetuned_bart_summarizer"):
        self.model_name = model_name
        self.saved_model_dir = saved_model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = None
        
        self.rouge_metric = evaluate.load("rouge")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            logger.logging.info("CUDA is detected, training is going to be run on GPU")
        else:
            logger.logging.info("training is going to be run on CPU")

        self.train_df, self.val_df, self.test_df = DataTransformer().preprocessed()

    def preprocess_data(self, examples):
        """Helper function for preprocessing datasets."""
        # Check for presence of 'conversation' and 'summary' keys
        try:
            if "conversation" not in examples or "summary" not in examples:
                logger.logging.info("Dataset examples must contain 'conversation' and 'summary' keys.")
            
            inputs = examples["conversation"]
            model_inputs = self.tokenizer(inputs, max_length=512, padding="max_length",truncation=True)
            
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(examples["summary"], max_length=150, padding="max_length", truncation=True)
            
            model_inputs["labels"] = labels["input_ids"]
        except Exception as e:
            raise exception.CustomException(e, sys) from e
        return model_inputs

    def tokenize(self):
        try:
            dataset = DatasetDict({
            "train": Dataset.from_pandas(self.train_df),
            "validation": Dataset.from_pandas(self.val_df),
            "test": Dataset.from_pandas(self.test_df),
            })
            logger.logging.info("Start tokenizing the dataset...")
            tokenized_dataset = dataset.map(
                self.preprocess_data, 
                batched=True, 
                remove_columns=[col for col in ["conversation", "summary"] if col in dataset["train"].column_names]
            )
        except Exception as e:
            raise exception.CustomException(e, sys) from e
        return tokenized_dataset

    def finetune(self, dataset_name="tweet_qa_tweetsumm", num_epochs=5, per_device_train_batch_size=2):
        """
        Fine-tunes the BART model on the specified dataset.
        """
        try:
            if os.path.exists(self.saved_model_dir) and len(os.listdir(self.saved_model_dir))>1:
                logger.logging.info(f"Fine-tuned model already exists at {self.saved_model_dir}. Loading it instead of fine-tuning.")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.saved_model_dir).to(self.device)
            
            else:
                tokenized_dataset = self.tokenize()
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

                #fine tune the model
                if not os.path.exists(f"{AppConfig.SUM_MODEL_DIR}/results"):
                    os.makedirs(f"{AppConfig.SUM_MODEL_DIR}/results")

                training_args = Seq2SeqTrainingArguments(
                    output_dir=f"{AppConfig.SUM_MODEL_DIR}/results",
                    eval_strategy="epoch",
                    learning_rate=2e-5,
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_train_batch_size,
                    weight_decay=0.01,
                    save_total_limit=3,
                    num_train_epochs=num_epochs,
                    predict_with_generate=True,
                    fp16=torch.cuda.is_available(),
                )

                trainer = Seq2SeqTrainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=tokenized_dataset["train"],
                    eval_dataset=tokenized_dataset["validation"],
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_metrics
                )

                time_start = time()
                logger.logging.info("Starting fine-tuning...")
                trainer.train()
                time_end = time()

                time_elapsed = time_end - time_start
                hours = int(time_elapsed // 3600)
                minutes = int((time_elapsed % 3600) // 60)
                seconds = time_elapsed % 60

                logger.logging.info(f"Training time for training the Bert Summarizer: {hours}h {minutes}m {seconds:.2f}s")
                
                # print evalution
                eval_results = trainer.evaluate()
                #logger.logging.info(f"Evaluation results: {eval_results}")
                print(f"Loss: {eval_results['eval_loss']}")
                for key, value in eval_results.items():
                    if key.startswith("rouge"):
                        print(f"{key}: {value:.2f}")


                #save the trained model for future use
                trainer.save_model(self.saved_model_dir)
                logger.logging.info(f"\nFine-tuning Bert Summarizer complete. Model saved to {self.saved_model_dir}")
        except Exception as e:
            raise exception.CustomException(e, sys) from e
        

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = self.rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        # Log ROUGE scores
        return {
        "rouge1": result["rouge1"] * 100,
        "rouge2": result["rouge2"] * 100,
        "rougeL": result["rougeL"] * 100,
        }
    
    def load_model(self):
        """
        Loads the fine-tuned or pre-trained model.
        """
        if self.model is not None:
            return
        
        if os.path.exists(self.saved_model_dir) and len(os.listdir(self.saved_model_dir))>1:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.saved_model_dir).to(self.device)            
            logger.logging.info("Loaded fine-tuned summarization model.")
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            logger.logging.info("Loaded pre-trained summarization model.")
    
    def get_summarization_pipeline(self):
        """Returns summarization pipeline for inference."""
        if self.model is None:
            self.load_model()
        
        return pipeline(
            "summarization", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            device=0 if torch.cuda.is_available() else -1
        )