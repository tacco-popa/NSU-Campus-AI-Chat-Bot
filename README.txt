##############################################################################
# NSU Campus AI Assistant
# Deepseek R1 1.5B


##############################################################################


1. Create virtual environment

python -m venv chatbot_e1

2. Activate the environment

chatbot_e1\Scripts\activate


3. Install Python Dependencies

pip install streamlit ollama PyMuPDF faiss-cpu numpy sentence-transformers

( install Ollama in your PC as well ) 


4. Pull the Deepseek R1 1.5B model

ollama pull deepseek-r1:1.5b


--------------------------------------------------------

5. Running the APP

streamlit run app.py

if it fails to run (unlikely) use a custom port.

streamlit run app.py --server.port 8502




Miscellaneous

evaluate.py used for calculating Similarity score
