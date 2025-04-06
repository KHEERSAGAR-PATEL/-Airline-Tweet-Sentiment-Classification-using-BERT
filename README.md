
---

# ✈️ BERT-based Airline Tweet Sentiment Analysis

Fine-tuning BERT for sentiment classification of tweets about airlines. Built with Hugging Face Transformers, PyTorch, and NLP preprocessing tools like spaCy and NLTK.

---

## 🚀 Project Overview

This project fine-tunes the `bert-base-uncased` model to classify tweets about U.S. airlines into **positive**, **neutral**, or **negative** sentiment categories. It demonstrates end-to-end NLP pipeline development using real-world data.

- 📊 Dataset: Airline Twitter Sentiment (from Kaggle)  
- 🧠 Model: Pre-trained BERT (`bert-base-uncased`)  
- 🧰 Tools: Hugging Face, PyTorch, spaCy, NLTK  
- 📈 Metrics: Accuracy, Precision, Recall, F1-score  
- 📉 Visualizations: Training curves, Confusion Matrix, Word frequencies  

---

## 📂 Dataset

- **Source**: [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)  
- **Classes**:  
  - `positive`  
  - `neutral`  
  - `negative`  
- **Columns used**: `text`, `airline_sentiment`

---

## 🧪 Pipeline Steps

1. **Data Loading**  
   Load and explore raw tweets from CSV.

2. **Preprocessing**  
   Clean, normalize, lemmatize tweets using spaCy & NLTK.

3. **EDA (Exploratory Data Analysis)**  
   Visualize word frequencies and sentiment distribution.

4. **Tokenization**  
   Use Hugging Face BERT tokenizer for encoding inputs.

5. **Dataset & DataLoader**  
   Prepare PyTorch datasets and loaders.

6. **Model Fine-tuning**  
   Load `bert-base-uncased` and fine-tune on our dataset using AdamW.

7. **Evaluation**  
   Assess model via classification report, confusion matrix, and loss visualization.

---

## 📊 Results

- **Num examples**: 2196  
- **Batch size**: 10  
- **Epoch**: 3  
- **Loss**: 2.8090  
- **Accuracy**: 0.8142  
- **Precision**: 0.8079  
- **Recall**: 0.8142  
- **F1 Score**: 0.8102  
- **Runtime**: 15.22 seconds  
- **Samples/sec**: 144.30  
- **Steps/sec**: 14.46  

### 📋 Full Classification Report

```
              precision    recall  f1-score   support

           0     0.8641    0.9055    0.8843      1439
           1     0.6294    0.5487    0.5863       421
           2     0.7913    0.7560    0.7732       336

    accuracy                         0.8142      2196
   macro avg     0.7616    0.7367    0.7479      2196
weighted avg     0.8079    0.8142    0.8102      2196
```

---

## 🧰 Tech Stack

- Python 3.x  
- Transformers (Hugging Face)  
- PyTorch  
- Pandas, Numpy, Matplotlib  
- NLTK, spaCy  
- scikit-learn  

---

## 📁 Folder Structure

```
bert-airline-sentiment/
│
├── data/                        # Dataset files
├── notebook/bert_sentiment.ipynb
├── models/                      # Saved model weights (optional)
├── visualizations/              # Loss curves, confusion matrix
└── README.md
```

---

## 📌 Future Work

- 🔮 Hyperparameter tuning with Optuna  
- 🌍 Streamlit web demo  
- 📦 Export model for Hugging Face Hub  
- ⚙️ Use distilBERT for faster inference  

---

## ✨ Author

Kheer Sagar Patel 
M.Tech in AI & ML, IIITDM Jabalpur  

---

## 💡 Inspiration

This project was built as part of an applied AI & NLP capstone, demonstrating how transformer models can be adapted to solve real-world business problems like sentiment tracking and social media monitoring.

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).

---
