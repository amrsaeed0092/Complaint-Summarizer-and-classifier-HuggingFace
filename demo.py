from flask import Flask, render_template, request
from config import AppConfig
from modules import logger, exception
from summarizer.BertSummarizer import BartSummarizer
from classifer.BartClassifier import ComplaintClassifier


app = Flask(__name__)

def summarize_and_classify_new_complaints(new_complaints_list):
    """
    Summarizes and classifies a list of new complaints.
    
    Args:
        new_complaints_list (list): A list of new complaint narratives (strings).
        
    Returns:
        list: A list of dictionaries, each containing the original complaint,
              its summary, and its classification.
    """
    candidate_labels = [
        "Credit Card",
        "Mortgage",
        "Bank Account or Service",
        "Money Transfer",
        "Prepaid Card",
        "Student Loan",
        "Vehicle Loan",
        "Consumer Loan",
        "Credit Reporting",
    ]
    
    try:
        # Load the summarization pipeline from the fine-tuned model
        summarizer_instance = BartSummarizer()
        summarization_pipeline = summarizer_instance.get_summarization_pipeline()
        
        # Instantiate the zero-shot classifier
        classifier_instance = ComplaintClassifier()
    except Exception as e:
        logger.logging.info("Error: Fine-tuned model not found. Please run `finetune_bart_oop.py` first.")
        return None

    logger.logging.info("Generating summaries and classifications for new complaints...")
    
    results = []
    
    for complaint in new_complaints_list:
        if len(complaint) > 100:
            # Step 1: Generate the summary
            summary = summarization_pipeline(
                complaint,
                max_length=150,
                min_length=30,
                length_penalty=2.0,
                num_beams=4,
                #padding="max_length",
                truncation=True
            )[0]['summary_text']
        else: 
            summary = complaint
            
        # Step 2: Classify the generated summary
        classification_result = classifier_instance.classify(summary, candidate_labels)
        
        # Extract the best label and its score
        predicted_label = classification_result['labels'][0]
        prediction_score = classification_result['scores'][0]
        
        results.append({
            "original_complaint": complaint,
            "summary": summary,
            "classification": predicted_label,
            "score": prediction_score
        })
        
    return complaint, summary, predicted_label, prediction_score


@app.route("/", methods=["GET", "POST"])
def index():
    dialog=[]
    if request.method == "POST":
        dialog.append ( request.form.get("dialog"))
        for item in dialog:
            if not item:
                return render_template("index.html", error="Please enter the customer/representative dialog here...")
            
        complaint, summary, predicted_label, prediction_score = summarize_and_classify_new_complaints(dialog)
        print(prediction_score)
        #return result
        return render_template("result.html", dialog=complaint, summary=summary, classification=predicted_label)
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=7777)
