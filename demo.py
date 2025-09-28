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
        summarizer_instance = BartSummarizer(model_name=AppConfig.SUMM_MODELNAME, saved_model_dir=AppConfig.SUM_MODEL_DIR)
        
        summarization_pipeline = summarizer_instance.get_summarization_pipeline()
        
        # Instantiate the zero-shot classifier
        classifier_instance = ComplaintClassifier(model_name=AppConfig.CLASSIFIER_MODELNAME)
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
                truncation=True
            )[0]['summary_text']
        else: 
            summary = complaint
        
        if not summary:
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
        
        print("results is : ",results)
    return complaint, summary, predicted_label, prediction_score


@app.route("/", methods=["GET", "POST"])
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        dialog_text = request.form.get("dialog", "").strip()
        if not dialog_text:
            return render_template("index.html", error="Please enter the customer/representative dialog here...")

        # summarize_and_classify expects a list of complaints
        complaint, summary, predicted_label, prediction_score = summarize_and_classify_new_complaints([dialog_text])

        return render_template(
            "result.html",
            dialog=complaint,
            summary=summary,
            classification=predicted_label
        )
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True, port=7777)
