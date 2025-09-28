# Complaint Summarizer and Classifier Using Transformers from Hugginh Face Hub 🚀  

## 📌 Project Summary  
This project builds an **AI-powered system** that automatically **summarizes** and **classifies** customer complaints.  

### ❓ Problem  
Manually processing customer complaints is time-consuming for businesses. Employees must read through numerous long submissions to understand the core issues and route them to the correct department. This process is inefficient and error-prone.  

### 💡 Solution  
We developed a **Natural Language Processing (NLP) solution** that leverages a fine-tuned **BART sequence-to-sequence model** to:  

- **Complaint Summarization**: Generate concise summaries of customer complaints, enabling staff to quickly grasp the key issues.  
- **Complaint Classification**: Automatically assign complaints to categories such as **billing**, **technical support**, or **shipping**, ensuring fast routing to the right team.  

---

## ⚙️ Methodology  

1. **Model Fine-Tuning**  
   - Pre-trained **BART model** fine-tuned on a custom dataset of customer complaints.  
   - Implemented using Hugging Face’s `Seq2SeqTrainer`.  

2. **Training**  
   - Trained for **5 epochs** on the dataset.  

3. **Evaluation**  
   - Performance tracked using **ROUGE metrics** to measure summarization quality.  
   - Validation loss and training loss monitored to detect overfitting.  

4. **Optimization Insights**  
   - Validation loss increased after the **first epoch**, while training loss decreased → clear **overfitting**.  
   - Future work: add **early stopping** and save the **best-performing checkpoint**.  

---

## 📊 Key Results  

- Best performance achieved at **epoch 1**:  
  - **ROUGE-1**: ~41%  
  - **ROUGE-L**: ~33%  

- Generated summaries were **coherent and concise**, greatly reducing manual review time.  
- Despite some overfitting, the model maintained **stable ROUGE performance** across validation.  

---

## 🛠️ Tech Stack  

- **Python 3.9+**  
- **Hugging Face Transformers**  
- **PyTorch**  
- **Datasets (Hugging Face)**  
- **Flask (for deployment UI)**  

---

## 🚀 How to Run  

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/Complaint-Summarizer-and-Classifier-HuggingFace.git
   cd Complaint-Summarizer-and-Classifier-HuggingFace

2. run python app.py
3. Once the model is trained, run demo.py for testing
