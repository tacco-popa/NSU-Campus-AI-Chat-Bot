# NSU Campus AI Chat Bot  
> *Empowering students with instant, accurate, and context-aware answers — powered by AI and Retrieval-Augmented Generation (RAG).*
> 
An AI-powered chatbot built for North South University’s SEPS students. Uses Retrieval-Augmented Generation (RAG), DeepSeek R1 LLM, and FAISS to deliver instant, accurate, and context-aware answers about courses, policies, and campus resources via an intuitive Streamlit UI.




---

## 📖 Introduction
Navigating university resources can be challenging, especially when reliable information is buried across multiple web pages.  

![NSU Banner](https://i.imgur.com/WL0mFbh.png)

At **North South University (NSU)**, students face:
- ❗ Confusing academic jargon  
- ⏳ Outdated or slow search results  
- ❌ Reliance on unofficial sources like Facebook groups → leading to misinformation  
- 📞 Delays due to long helpline wait times or unavailable staff  

The **NSU Campus AI Assistant** solves these problems by:
- Simplifying complex academic terms and policies  
- Delivering **verified, real-time answers** 24/7  
- Reducing misinformation with **trusted data sources**  
- Guiding students through NSU’s SEPS resources seamlessly  

> Built with **DeepSeek R1**, **FAISS**, **Sentence Transformers**, and **Streamlit**, this chatbot transforms how students interact with academic data.

---

## 🎯 Problem vs Solution
![NSU Banner](https://i.imgur.com/ARs6Aou.png)

| **Problem**                           | **Solution** |
|--------------------------------------|--------------|
| Students waste time manually searching websites. | ⚡ Instant AI-driven search through a chatbot |
| Outdated or incorrect information from unofficial sources. | ✅ Verified, real-time data retrieval |
| Complex jargon causes confusion. | 🧠 Simplified explanations in natural language |
| Long wait times for support staff. | 🤖 24/7 automated assistance |

---

## 🧠 System Overview
![System Architecture](docs/system-architecture.png)

The project integrates **advanced AI techniques** to deliver a seamless experience:
- **Retrieval-Augmented Generation (RAG)**  
- **Vector-based Semantic Search** (FAISS)  
- **Sentence Transformers** for dense embeddings  
- **DeepSeek R1 LLM** for natural, context-aware conversations  
- **Streamlit UI** for an intuitive interface

---

## ✨ Features
- 📚 **Domain-Specific Knowledge Base** — tailored to NSU SEPS students  
- ⚡ **Lightning-Fast Semantic Search** powered by FAISS  
- 🖥️ **Beautiful Streamlit UI** with NSU-themed design and custom CSS  
- 📝 **Cached Data** for performance optimization  
- 🔧 **Customizable Settings** (temperature, token limits, model behavior)  
- 🗂 **PDF Support** — ingest and parse academic documents

---

## 🧰 Tech Stack
| Component        | Technology Used            |
|------------------|----------------------------|
| **LLM Runtime**  | Ollama + DeepSeek R1 (1.5B)|
| **Vector Search**| FAISS                       |
| **Embeddings**   | Sentence Transformers (bge-small-en-v1.5) |
| **Frontend**     | Streamlit + Custom CSS      |
| **Backend**      | Python, JSON caching        |

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/tacco-popa/NSU-Campus-AI-Chat-Bot.git
cd NSU-Campus-AI-Chat-Bot
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Prepare LLM Install Ollama
 and pull the DeepSeek R1 model:
```bash
ollama pull deepseek-r1:1.5b
```

### 5. Run the Application
```bash
streamlit run app.py
```

---

## 📝 Example Queries
| **Query** | **Expected Response** |
|------------|-----------------------|
| "When was SEPS founded?" | SEPS started as SEAS in 1993 and was renamed in 2014. |
| "How many students are enrolled?" | Over 7,500 students are enrolled in SEPS. |
| "What labs does CEE have?" | 6 testing labs and 1 drawing lab currently available. |

> These queries were part of our BLEU score evaluation, with an **average similarity score of 0.633**.

---

## 📊 Evaluation Metrics
The project was tested on 12 representative queries.  
**BLEU Score Range:** 0.294 – 0.846 (average: 0.633)

![BLEU Score Graph](docs/bleu-score.png)

---

## 📅 Project Timeline
| Phase | Description | Weeks |
|-------|-------------|-------|
| 1 | Research & Planning | 1-2 |
| 2 | Model Selection & Experiments | 3-4 |
| 3 | Data Collection & Preprocessing | 5-6 |
| 4 | RAG Pipeline & Interface Development | 7-8 |
| 5 | Fine-Tuning & Evaluation | 9-10 |
| 6 | Deployment & Refinement | 11-12 |
| 7 | Final Documentation & Presentation | 13-14 |

---

## 🔮 Future Scope
- Integration of **real-time announcements** and faculty-specific data  
- **RLHF-based fine-tuning** for complex, context-heavy queries  
- Full **automation of data ingestion pipelines**  
- Expanded coverage for other NSU departments  

---

## 🤝 Contributing
Contributions are welcome!  
See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 🛡 License
This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.  

You are free to use, modify, and share this project **for non-commercial purposes only**, with proper attribution.  
Commercial use of this code is **strictly prohibited**.

Full license: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)

