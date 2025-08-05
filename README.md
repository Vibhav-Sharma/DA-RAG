# 📚 Dynamic Wikipedia-RAG System using BART + Wikipedia + Topic-Aware Clarification

This project implements a **Dynamic Retrieval-Augmented Generation (RAG)** system that:
- Retrieves Wikipedia data dynamically based on user queries
- Extracts topic headings for clarification
- Allows follow-up responses based on subtopics
- Generates context-aware answers using `facebook/bart-large-cnn`

---

## 🔧 Architecture Overview

> **Model:** SentenceTransformer for embedding retrieval  
> **Retriever:** Dynamic Wikipedia fetch with topic-based filtering  
> **Generator:** BART (facebook/bart-large-cnn)  
> **Clarification Mechanism:** Based on Wikipedia topic headings  
> **Comparison Models:** t5-base, flan-t5-base

![Architecture Diagram]![Uploading ChatGPT Image Aug 5, 2025, 08_49_15 PM.png…]()


---

## 📌 Features

- 🔍 Wikipedia-powered semantic document retriever  
- 🧠 Sub-topic clarification before generation  
- ✏️ Follow-up refinement using subtopic index  
- 📊 Evaluation & benchmarking with ROUGE, BLEU, BERTScore, METEOR

---

## 📈 Model Evaluation

### 🔬 Average Generation Time (seconds)
![Generation Time Graph]![WhatsApp Image 2025-07-23 at 22 45 37_d636adb6](https://github.com/user-attachments/assets/fd33c51e-db69-4391-bf27-a4c22749c0f8)


### 🧪 ROUGE-L Score Comparison
![ROUGE Score Graph]![WhatsApp Image 2025-07-23 at 22 45 50_46ac1665](https://github.com/user-attachments/assets/78318013-dfeb-4b5d-ad74-151d2aadef1f)


### 🔤 BLEU Score Comparison
![BLEU Score Graph]![WhatsApp Image 2025-07-23 at 23 19 03_704487c3](https://github.com/user-attachments/assets/3f31652e-863a-40d6-a18c-aafd4999a9c5)



---

## 📊 Metric Table

| Metric       | facebook/bart-large-cnn | t5-base | flan-t5-base |
|--------------|--------------------------|---------|--------------|
| ROUGE-1      | 0.6285                   | 0.3889  | 0.3428       |
| ROUGE-2      | 0.4242                   | 0.0588  | 0.1212       |
| ROUGE-L      | 0.6285                   | 0.2777  | 0.2857       |
| BLEU         | 0.83xx                   | ~0      | ~0           |
| BERTScore F1 | 0.4393                   | 0.1220  | 0.0260       |
| Time (s)     | 25.3                     | 11.6    | 8.4          |

---

## 🧩 Use Case Flow

1. **User inputs a vague query**
2. **System fetches Wikipedia pages and extracts topics**
3. **Clarification is prompted from the user (optional)**
4. **Sub-topic index input triggers refined query**
5. **Response generated from selected model**
6. **Metrics logged and compared**

---

## 🧪 Future Scope

- [ ] Add support for PDF/Text file sources  
- [ ] Add LoRA-based model compression  
- [ ] Multi-model switch for user control  
- [ ] Live chat with response streaming  

---

## 📂 Run Locally

```bash
git clone https://github.com/yourrepo/wiki-rag
cd wiki-rag
pip install -r requirements.txt
python app.py
