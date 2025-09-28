import torch
from transformers import pipeline
from config import AppConfig
from modules import exception, logger


class ComplaintClassifier:
    """
    A class to perform zero-shot classification on text data.
    Uses a pre-trained model fine-tuned on a Natural Language Inference (NLI) task.
    """
    def __init__(self, model_name="facebook/bart-large-mnli"):
        self.model_name = model_name
        self.classifier = None
        
    def load_classifier(self):
        """Loads the zero-shot classification pipeline."""
        if self.classifier is None:
            logger.logging.info("Loading zero-shot classification pipeline...")
            self.classifier = pipeline(
                "zero-shot-classification", 
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.logging.info("Zero-shot classifier loaded.")
            
    def classify(self, text, candidate_labels):
        """
        Classifies the given text into one of the candidate labels.
        
        Args:
            text (str): The text to classify.
            candidate_labels (list): A list of possible classification labels.
            
        Returns:
            dict: The classification result.
        """
        self.load_classifier()
        # Ensure input is a list, as the pipeline expects it
        return self.classifier(text, candidate_labels)

'''if __name__ == "__main__":
    # Example usage for the classifier
    classifier = ComplaintClassifier()
    sample_text = "I was charged an incorrect fee on my credit card."
    labels = ["Credit Card", "Mortgage", "Bank Account or Service"]
    result = classifier.classify(sample_text, labels)
    print(result)
'''