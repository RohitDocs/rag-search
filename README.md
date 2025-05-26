# Lightweight RAG Bot with Gemini 1.5

A lightweight Retrieval-Augmented Generation (RAG) app built with:
- FAISS for vector search
- Sentence Transformers for embedding
- Gemini 1.5 (Google Generative AI) for answering questions
- Streamlit for the user interface

---

## 🚀 Setup Instructions

1. **Clone/Download the project.**
   ```bash
   git clone https://github.com/your-repo/lightweight-rag-bot.git
   cd lightweight-rag-bot
   ```

2. **Create and activate a virtual environment.**
   ```bash
   python -m venv .venv
   # Activate:
   # Windows Powershell: & .\.venv\Scripts\Activate.ps1
   # Windows CMD: .venv\Scripts\activate
   # Linux/macOS: source .venv/bin/activate
   ```

3. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file** with your Gemini API key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Run the Streamlit app.**
   ```bash
   streamlit run app.py
   ```

6. **Optional: Test the setup.**
   Run the following command to ensure all dependencies are installed and the app is functional:
   ```bash
   python -m unittest discover tests
   ```

---

## 📦 Folder Structure

```text
.
├── app.py                      # Main Streamlit app
├── data/
│   ├── faiss.index             # FAISS vector index
│   └── gdpr_articles.json      # Retrieved document contents
├── tests/                      # Unit tests for the app
├── .env                        # API keys (not shared publicly)
├── .venv/                      # Python virtual environment
├── requirements.txt            # Python dependencies
└── README.md                   # Project instructions
```

---

## 🛠 Features

- Embeds user queries and retrieves similar documents using FAISS.
- Generates contextually grounded answers using Gemini 1.5.
- Lightweight and fast — no heavy server setup needed.
- Easy to extend for more advanced RAG functionalities.

---

## 🧪 Example Queries

- "When does GDPR apply outside Europe?"
- "What are the key principles of GDPR?"
- "What is the right to be forgotten under GDPR?"
- "What are the legal grounds for processing personal data?"

---

## 📋 Requirements

- Python 3.8+
- FAISS (`faiss-cpu`)
- Sentence Transformers
- Streamlit
- Google Generative AI (`google-generativeai`)
- Python Dotenv
- Certifi (for SSL certificate validation)
- NumPy (for numerical operations)

---

## ✅ Notes

- Make sure your `.env` file is correctly configured.
- Gemini API key must have access to the `gemini-1.5-pro` or `gemini-1.5-flash` models.
- Default model used: `gemini-1.5-pro`.
- The `tests/` folder contains unit tests to validate the app's functionality.

---

Built with ❤️ to make document Q&A lightweight, fast, and effective.
