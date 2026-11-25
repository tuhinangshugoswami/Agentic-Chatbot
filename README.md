# Agentic Chatbot using LangGraph and Streamlit

I built this project to learn how agentic AI systems work. The idea was to go beyond a simple chatbot and build something that can actually use tools — like searching the web, doing calculations, fetching stock prices, and answering questions from a PDF you upload.

The agent is built using LangGraph's ReAct loop, where the LLM decides which tool to call based on your question. Conversations are saved to SQLite so memory is preserved across sessions.

---

## What it can do

- Search the web using DuckDuckGo
- Fetch live stock prices (e.g. AAPL, TSLA) via Alpha Vantage
- Do basic calculations (add, subtract, multiply, divide)
- Answer questions from an uploaded PDF using RAG (FAISS + FastEmbed)
- Remember conversation history across multiple sessions using SQLite

---

## Tech used

- **LangGraph** — for building the ReAct agent and managing the graph
- **Streamlit** — for the frontend UI
- **LangChain** — for tools, loaders, and LLM integration
- **FAISS + FastEmbed** — for vector search on uploaded PDFs
- **SQLite** — for persisting chat memory across sessions
- **OpenRouter** — to access the LLM (free tier)

---

## How to run

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# then open .env and paste your OpenRouter API key

# 5. Run
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Project structure

```
backend.py       # agent logic, tools, RAG, graph setup
app.py           # streamlit UI
requirements.txt
.env.example
```

---

## Notes

- Get a free API key from [openrouter.ai](https://openrouter.ai)
- On first run, FastEmbed downloads the embedding model (~40MB) so you need internet
- The SQLite database (`chatbotdb.db`) is created locally and is not pushed to GitHub
