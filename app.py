###########################################
# NSU Campus AI Assistant
# Deepseek R1 1.5B
# Tanvir Bin Zahid

###########################################

import streamlit as st
import ollama
import re
import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer


# Set the page config to start with the sidebar collapsed
st.set_page_config(initial_sidebar_state="collapsed")


###########################################
# PDF Extraction and RAG Functions with Caching
###########################################


def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            raw_text = page.get_text()

            cleaned_text = re.sub(r'\n\s*\n+', '\n', raw_text)
            cleaned_text = re.sub(r'Page \d+', '', cleaned_text)
            text += cleaned_text + "\n"
    return text.strip()

def load_all_pdfs(folder_path):
    """Load all PDFs using cached JSON if available and up-to-date."""
    json_path = os.path.join(folder_path, "pdf_data.json")
    current_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(".pdf")]

    # Try to load cached data if exists
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                saved_data = json.load(f)

            # Validate cached data
            needs_refresh = False
            saved_files = {entry["filename"]: entry for entry in saved_data}

            # Check for file changes
            current_set = set(current_files)
            saved_set = set(saved_files.keys())

            # Check for added/removed files
            if current_set != saved_set:
                needs_refresh = True
            else:
                # Check modification times
                for filename in current_files:
                    file_path = os.path.join(folder_path, filename)
                    current_mtime = os.path.getmtime(file_path)
                    if saved_files[filename]["last_modified"] < current_mtime:
                        needs_refresh = True
                        break

            if not needs_refresh:
                return [{"filename": entry["filename"], "text": entry["text"]}
                        for entry in saved_data]

        except (json.JSONDecodeError, KeyError):
            pass  # Invalid cache, regenerate

    # Process PDFs and cache results
    docs = []
    for filename in current_files:
        path = os.path.join(folder_path, filename)
        text = extract_text_from_pdf(path)
        docs.append({
            "filename": filename,
            "text": text,
            "last_modified": os.path.getmtime(path)
        })

    # Save to cache
    with open(json_path, "w") as f:
        json.dump(docs, f)

    return [{"filename": doc["filename"], "text": doc["text"]} for doc in docs]


def split_text(text, max_length=5000):
    """Split text into chunks of specified maximum length."""
    sentences = text.split('\n')
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks



# Initialize the SentenceTransformer model for embeddings
model = SentenceTransformer('bge-small-en-v1.5')

#############################################################################################
def build_vector_store(docs):
    """Build a FAISS vector store from document chunks."""
    all_chunks = []
    metadata = []
    for doc in docs:
        chunks = split_text(doc["text"])
        all_chunks.extend(chunks)
        metadata.extend([{"filename": doc["filename"]}] * len(chunks))
    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, all_chunks, metadata


###################################################################################################################

def retrieve_context(query, index, chunks, top_k=5):
    """Retrieve relevant context from the vector store."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return "\n\n".join(retrieved)


###########################################
# Chatbot Interface and Styling
###########################################

# Custom CSS styling
# Custom CSS styling
st.markdown("""
<style>
/* Dark theme with background image */
.stApp {
    background-image: url('https://i.imgur.com/wC9Qhc6.png');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    color: #ffffff;
}
/* Chat messages */
.stChatMessage {
    padding: 1.5rem;
    border-radius: 1.25rem;
    margin: 1rem 0;
    max-width: 80%;
    position: relative;
    background: rgba(30, 41, 59, 0.8); /* Semi-transparent background for messages */
}
/* User messages */
[data-testid="stChatMessage"][aria-label="user"] {
    background: rgba(30, 41, 59, 0.9); /* Slightly more opaque for user messages */
    margin-left: auto;
    border: 1px solid #334155;
}
/* AI messages */
[data-testid="stChatMessage"][aria-label="AI"] {
    background: rgba(30, 58, 138, 0.9); /* Slightly more opaque for AI messages */
    margin-right: auto;
    animation: slideIn 0.3s ease;
}
/* Input box */
.stTextInput input {
    background: rgba(30, 41, 59, 0.8) !important;
    color: white !important;
    border-radius: 1rem !important;
    padding: 1rem !important;
    border: 1px solid #334155 !important;
}
/* Animations */
@keyframes slideIn {
    from { transform: translateY(20px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}
/* Header */
.header {
    background: linear-gradient(90deg, rgba(59, 130, 246, 0.8) 0%, rgba(99, 102, 241, 0.8) 100%);
    padding: 1.5rem;
    margin-bottom: 2rem;
    border-radius: 0 0 1rem 1rem;
}
/* Typing animation */
.typing-animation {
    display: flex;
    align-items: center;
    height: 17px;
}
.dot {
    width: 6px;
    height: 6px;
    margin: 0 2px;
    background: #fff;
    border-radius: 50%;
    animation: typing 1.4s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes typing {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-5px); }
}
/* Reasoning text */
.reasoning {
    color: #94a3b8;
    font-style: italic;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1 style="color:white; margin:0;">NSU Campus AI Assistant</h1>
    <p style="color:white; margin:0; opacity:0.9;">Powered by AI</p>
</div>
""", unsafe_allow_html=True)

###########################################
# Session State Initialization
###########################################

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I assist you today? 🚀"}
    ]
    st.session_state.model_config = {
        "temperature": 0.2,
        "top_p": 0.9,
        "max_tokens": 912,
        "repeat_penalty": 1.1
    }
    st.session_state.show_thinking = True
    st.session_state.show_reasoning = True

# Load saved settings
if os.path.exists("settings.json"):
    with open("settings.json", "r") as f:
        saved_settings = json.load(f)
        st.session_state.show_thinking = saved_settings.get("show_thinking", st.session_state.show_thinking)
        st.session_state.show_reasoning = saved_settings.get("show_reasoning", st.session_state.show_reasoning)
        st.session_state.model_config["temperature"] = saved_settings.get("temperature",
                                                                          st.session_state.model_config["temperature"])
        st.session_state.model_config["max_tokens"] = saved_settings.get("max_tokens",
                                                                         st.session_state.model_config["max_tokens"])

# Initialize vector store with cached PDF loading
if "vector_index" not in st.session_state:
    pdf_folder = "./pdfs"
    if os.path.exists(pdf_folder):
        docs = load_all_pdfs(pdf_folder)
        vector_index, chunks, metadatas = build_vector_store(docs)
        st.session_state.vector_index = vector_index
        st.session_state.chunks = chunks
        st.session_state.metadatas = metadatas
    else:
        st.session_state.vector_index = None
        st.session_state.chunks = None
        st.session_state.metadatas = None

###########################################
# Sidebar Controls
###########################################

with st.sidebar:
    st.header("Controls")
    last_is_ai = len(st.session_state.messages) > 0 and st.session_state.messages[-1]["role"] == "assistant"
    if st.button("🔄 Regenerate Last Response", disabled=not last_is_ai):
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()
            st.session_state.regenerate = True

    st.subheader("Model Settings")
    st.toggle("Show Thinking Animation", key="show_thinking")
    st.toggle("Show AI Reasoning", key="show_reasoning")
    st.session_state.model_config["temperature"] = st.slider(
        "Temperature", 0.0, 1.0, st.session_state.model_config["temperature"], 0.1
    )
    st.session_state.model_config["max_tokens"] = st.slider(
        "Max Tokens", 128, 1024, st.session_state.model_config["max_tokens"], 128
    )


    def save_settings():
        settings = {
            "show_thinking": st.session_state.show_thinking,
            "show_reasoning": st.session_state.show_reasoning,
            "temperature": st.session_state.model_config["temperature"],
            "max_tokens": st.session_state.model_config["max_tokens"]
        }
        with open("settings.json", "w") as f:
            json.dump(settings, f)
        st.sidebar.success("Settings saved!")


    if st.button("💾 Save Settings"):
        save_settings()


    # Function to reload all PDFs and update JSON (super experimental) not sure if this helps or not, in theory JSON should help
    def recalculate_pdf_data():
        pdf_folder = "./pdfs"  # Root directory folder
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

        if not pdf_files:
            st.sidebar.error("No PDFs found in 'pdfs' folder.")
            return


        pdf_data = {"files": pdf_files}

        with open("pdf_data.json", "w") as json_file:
            json.dump(pdf_data, json_file, indent=4)

        st.sidebar.success("PDF data recalculated!")


    # Sidebar UI
    st.sidebar.header("Options")
    if st.sidebar.button("♻️ Recalculate PDF Data"):
        with st.spinner("Processing PDFs..."):
            recalculate_pdf_data()

# Avatars
user_avatar = "👤"
ai_avatar = "🤖"


###########################################
# Chat Functions
###########################################

def parse_response(response):
    """Extract reasoning and content from response using <think> tags."""
    match = re.search(r'<think>(.*?)</think>(.*)', response, re.DOTALL)
    if match:
        return {
            "reasoning": match.group(1).strip(),
            "content": match.group(2).strip()
        }
    return {"reasoning": "", "content": response}


def display_response(parsed, placeholder):
    """Display response with optional reasoning."""
    final_display = []
    if st.session_state.show_reasoning and parsed["reasoning"]:
        final_display.append(f"<div class='reasoning'>🤔 {parsed['reasoning']}</div>")
    final_display.append(parsed["content"])
    placeholder.markdown("\n".join(final_display), unsafe_allow_html=True)


def generate_response():
    """Generate and display AI response with RAG context."""
    user_prompt = st.session_state.messages[-1]["content"]
    retrieved_context = ""
    if st.session_state.vector_index is not None and st.session_state.chunks is not None:
        retrieved_context = retrieve_context(user_prompt, st.session_state.vector_index, st.session_state.chunks)

    system_prompt = f"""
         Use the following retrieved context to answer the query accurately:
         {retrieved_context}

         Try to always cite information from the documents. If unsure, say 'I don’t have enough information to answer this.'
         """

    # This helps (somewhat) to avoid hallucination

    augmented_messages = []
    if system_prompt:
        augmented_messages.append({"role": "system", "content": system_prompt})
    augmented_messages.extend(st.session_state.messages)

    with st.chat_message("assistant", avatar=ai_avatar):
        response_placeholder = st.empty()
        if st.session_state.show_thinking:
            response_placeholder.markdown("""
            <div style="display: flex; align-items: center; gap: 0.5rem">
                <div class="typing-animation">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        full_response = ""
        for chunk in ollama.chat(
                model="deepseek-r1:1.5b",
                messages=augmented_messages,
                stream=True,
                options={
                    "temperature": st.session_state.model_config["temperature"],
                    "top_p": st.session_state.model_config["top_p"],
                    "num_predict": st.session_state.model_config["max_tokens"],
                    "repeat_penalty": st.session_state.model_config["repeat_penalty"]
                }
        ):
            full_response += chunk["message"]["content"]
            cursor = "▌" if not st.session_state.show_thinking else ""
            response_placeholder.markdown(full_response + cursor)

        parsed = parse_response(full_response)
        message = {"role": "assistant", "content": parsed["content"], "reasoning": parsed["reasoning"]}
        st.session_state.messages.append(message)
        display_response(parsed, response_placeholder)

def is_response_incomplete(response):
    """Check if response appears incomplete."""
    response = response.strip()
    return response and response[-1] not in [".", "!", "?", "\"", "'"]

#Streamlit interface gets buggy implementing CHAT GPT like interface, this may or may not work. Finger Crossed

def continue_response():
    """Continue the last assistant response."""
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
        last_assistant = st.session_state.messages.pop()
        st.session_state.messages.append({"role": "user", "content": "Please continue your previous answer."})
        generate_response()
        new_assistant = st.session_state.messages.pop()
        combined_content = last_assistant["content"].strip() + "\n" + new_assistant["content"].strip()
        combined_reasoning = (last_assistant.get("reasoning", "").strip() + "\n" + new_assistant.get("reasoning",
                                                                                                     "").strip()).strip()
        st.session_state.messages.append({
            "role": "assistant",
            "content": combined_content,
            "reasoning": combined_reasoning
        })

#Streamlit interface gets buggy implementing CHAT GPT like interface, this may or may not work. Finger Crossed
###########################################
# Chat History Display
###########################################

for message in st.session_state.messages:
    role = "user" if message["role"] == "user" else "AI"
    with st.chat_message(role, avatar=user_avatar if role == "user" else ai_avatar):
        reasoning = message.get("reasoning", "")
        content = message.get("content", "")
        if st.session_state.show_reasoning and reasoning:
            st.markdown(f"<div class='reasoning'>🤔 {reasoning}</div>{content}", unsafe_allow_html=True)
        else:
            st.markdown(content)

if hasattr(st.session_state, "regenerate") and st.session_state.regenerate:
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        generate_response()
    st.session_state.regenerate = False

###########################################
# User Input Handling
###########################################

if prompt := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(prompt)
    generate_response()

