import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
import os
import ssl
import certifi
import asyncio
import logging
import requests

logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY is missing in .env.")
    st.stop()

try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    st.error(f"❌ Failed to configure Gemini API: {e}")
    st.stop()

faiss_index_path = os.path.abspath("data/faiss.index")
gdpr_articles_path = os.path.abspath("data/gdpr_articles.json")

@st.cache_resource(ttl=3600)
def load_faiss_index():
    return faiss.read_index(faiss_index_path)

@st.cache_resource(ttl=3600)
def load_sources():
    with open(gdpr_articles_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(ttl=3600)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def search_query(index, embedder, query, top_k=5):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return I[0]

def build_context_prompt(query):
    history = st.session_state.get("conversation_history", [])
    context = "\n".join([f"User: {q}\nBot: {a}" for q, a in history])
    return f"{context}\nUser: {query}\nBot:"

def ask_gemini(chunks, query):
    context = "\n\n".join(chunks)
    prompt = f"Using the following document parts, answer the question:\n\n{context}\n\n{build_context_prompt(query)}"
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    return response.text

def ask_claude(chunks, query, model_name="claude-3-haiku-20240307"):
    if not CLAUDE_API_KEY:
        return "❌ Claude API key missing. Please set it in .env."

    context = "\n\n".join(chunks)
    prompt = f"Using the following document parts, answer the question:\n\n{context}\n\n{build_context_prompt(query)}"

    headers = {
        "x-api-key": CLAUDE_API_KEY,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name,
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        content = response_json.get('content', [])
        if content and isinstance(content, list):
            return content[0].get('text', 'No text found')
        return "❌ Unexpected response format from Claude API."
    except Exception as e:
        logging.error(f"Claude API error: {e}")
        return f"❌ Claude API error: {e}"

def generate_follow_up_questions(answer):
    prompt = f"Based on this answer, suggest 2-3 concise follow-up questions:\n\n{answer}\n\nList them as bullet points."
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)
    return [line.strip("• ").strip() for line in response.text.splitlines() if line.strip()]

st.title("📚 Doc Agent for GDPR")

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

model_provider = st.selectbox("Select AI Provider:", ["Gemini", "Claude"])
if model_provider == "Claude":
    model_name = st.selectbox("Select Claude Model:", ["claude-3-haiku-20240307", "claude-3-opus-20240229"])
else:
    model_name = "gemini-1.5-pro"

query = st.text_input("Enter your question:")

if query and query.strip():
    with st.spinner("Searching documents..."):
        index = load_faiss_index()
        sources = load_sources()
        embedder = load_embedder()
        indices = search_query(index, embedder, query)
        selected_chunks = [sources[i]['content'] for i in indices if i < len(sources)]

    if not selected_chunks:
        st.error("❌ No relevant information found.")
    else:
        with st.spinner("Generating answer..."):
            if model_provider == "Gemini":
                answer = ask_gemini(selected_chunks, query)
            else:
                answer = ask_claude(selected_chunks, query, model_name=model_name)

        st.session_state.conversation_history.append((query, answer))

        st.markdown("### 📝 Answer:")
        st.write(answer)

        with st.spinner("Generating follow-up suggestions..."):
            follow_ups = generate_follow_up_questions(answer)
            if follow_ups:
                st.markdown("### 📌 Suggested Follow-ups:")
                for follow_up in follow_ups:
                    if st.button(follow_up):
                        st.experimental_set_query_params()
                        st.session_state["query_override"] = follow_up
