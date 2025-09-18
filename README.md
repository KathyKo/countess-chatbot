# countess-chatbot

# Countess RAG Chatbot

[![Python](https://img.shields.io/badge/python+-blue.svg)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/streamlit-orange.svg)](https://streamlit.io/)  
[![FAISS](https://img.shields.io/badge/faiss-lightgrey.svg)](https://github.com/facebookresearch/faiss)  
[![Ollama](https://img.shields.io/badge/ollama-green.svg)](https://ollama.com/)


A local AI-powered chatbot for **Countess bedding products**, using **LangChain**, **FAISS**, **Ollama LLM**, and **Ollama embeddings**.  
Supports **product-specific question answering**, **document retrieval**, and **multi-turn conversations**.

---

## Features

- Answer questions about Countess products:
  - Tencel™ (天絲)
  - Tencel-Cotton (天絲棉)
  - Modal Cotton (莫代爾棉)
  - Mercerized Cotton (絲光棉)
  - Brand & Company info
- Context-aware responses using **FAISS vector search**.
- Handles product ambiguity (e.g., 天絲 vs 天絲棉).
- Multi-turn conversation memory.
- Streamlit-based interface for local usage.

---

## Requirements

- Python   
- Streamlit  
- LangChain  
- LangGraph  
- FAISS  
- Ollama LLM Model (qwen3:8b)
- Ollama Embedding Model (nomic-embed-text)
- Dependencies listed in `requirements.txt`

---

## Usage

Start the chatbot:

```bash
streamlit run src/app.py
```

* Enter questions in the chat box.
* Chatbot retrieves relevant documents and provides answers.
* Supports multi-turn conversations: previous messages are remembered.
* To end a session, simply close the Streamlit tab or stop the server (`Ctrl+C`).

### Example Questions & Sample Answers

* **Question:** "莫代爾棉的特性有哪些？"  
  **Answer:**  
  - **材質組成:** 100% 莫代爾纖維  
  - **柔軟與彈性:** 改善純棉略硬質感，觸感柔軟  
  - **透氣與吸濕:** 保留天然透氣性，保持乾爽  
  - **色澤飽滿:** 染色均勻，布面色澤鮮亮且不易褪色  
  - **紡織技術:** 緊密賽絡紡，布料表面平滑、毛羽少，強度高、耐磨性佳  
  - **生產工藝:** 數位印花，色彩鮮豔，環保節水  
  - **設計亮點:** 窄幅布設計與拼接工藝，提升視覺層次與質感  
  - **注意事項:** 使用洗衣袋洗滌，減少摩擦以預防原纖化

* **Question:** "Countess 何時成立的？"  
  **Answer:**  
  Countess 於民國 67 年 (1978 年) 成立。

* **Question:** "Countess 臉書?"  
  **Answer:**  
  Countess 的 Facebook 粉絲專頁為: [https://www.facebook.com/CountessDesignStudio/](https://www.facebook.com/CountessDesignStudio/)


---

## File Structure

```
agentic_rag_bot/
├─ data/                  # Source documents (DOCX)
├─ faiss_index/           # Generated FAISS vector store
├─ src/
│  ├─ main.py            # Workflow and RAG logic
│  ├─ ingestion.py        # Document ingestion
│  └─ app.py              # Streamlit chatbot interface
├─ .env                   # Environment variables
├─ requirements.txt
└─ README.md
```

---

## Notes

* Ensure the FAISS vector store exists before running the app.
* Chat history is stored in memory for multi-turn conversations.
* The bot automatically removes `<think>` blocks for clean responses.
* Works locally; no external cloud service is required.

```
