

## Product Reviews Analysis with NLP

This project focuses on analyzing customer product reviews using Natural Language Processing techniques. It includes sentiment classification, category clustering, and review summarization using transformer-based models.

---

## 🚀 Overview  

This project transforms raw customer feedback into actionable business intelligence. By using advanced Deep Learning and Transformer models, it goes beyond simple word counts to understand the true sentiment and core themes within thousands of reviews. It helps businesses quickly identify what customers love and where they struggle without reading every single comment.

---


## 🛠️ Detailed Workflow
Sentiment Classification (BERT):
Fine-tuned a BERT-base model to capture the nuances of customer language, achieving high accuracy in distinguishing between positive, neutral, and negative feedback.

Product Clustering (Embeddings):
Utilized Sentence-BERT (SBERT) to create vector representations of products, allowing for the automatic grouping of similar items even when they don't share the same keywords.

Summarization (BART & GPT):
Implemented a dual-approach to summarization. BART was used for concise technical summaries, while GPT-3.5 was leveraged for more natural, human-like summaries of top pros and cons.

---

## Features

- **Binary Sentiment Classification**: Fine-tuned a BERT model to classify customer reviews as **Positive** or **Negative**.
- **Multi-Class Sentiment Classification**: Extended the model to classify reviews into **Positive**, **Neutral**, and **Negative** sentiments.
- **Product Clustering**: Used sentence embeddings and clustering methods to group similar products.
- **Summarization**: Generated summaries for each category using BART and GPT-3.5, highlighting top products and common complaints.

---

## Technologies Used

- Python, Pandas, TensorFlow
- Hugging Face Transformers (BERT, BART)
- SentenceTransformers
- Scikit-learn
- OpenAI API (GPT-3.5)

---

## 📸 Application Preview


### Single Review Analysis



<img width="1000" height="550" alt="pos" src="https://github.com/user-attachments/assets/f6fe7f28-5f9e-4bd3-8a23-0e9b48d25639" />

---

<img width="1000" height="550" alt="neg" src="https://github.com/user-attachments/assets/72901472-1ceb-4a4c-ad54-943793fa8022" />

---

### Batch Review Analysis

<img width="1000" height="550" alt="pross" src="https://github.com/user-attachments/assets/ee7a1813-1c86-4655-aebe-160c8bc75416" />

---
<img width="1000" height="550" alt="chart" src="https://github.com/user-attachments/assets/8a02f239-c9f7-419b-8a9e-e50360e39447" />

---
<img width="1000" height="550" alt="table" src="https://github.com/user-attachments/assets/73357047-259b-4928-9d0c-18744a3671ac" />


---

 For model file(tf_model.h5): 

https://drive.google.com/file/d/1P5Yjh3FhkPnCGLqNnlYRQ6AY3ebuf8ss/view?usp=sharing




