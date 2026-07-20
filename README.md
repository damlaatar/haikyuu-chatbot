# Haikyuu RAG Chatbot

A local FastAPI + React chat app that answers Haikyuu questions with retrieval over the text files in `haikyuu_db`.

The backend loads `unsloth/Llama-3.2-3B-Instruct`, embeds the text database with `sentence-transformers/all-MiniLM-L6-v2`, stores the local vector index in Chroma, and streams answers to the Vite UI.

## Project Layout

```text
.
├── haikyuu_rag_chat_api.py      # Main FastAPI RAG backend
├── run_haikyuu_chat.py          # Starts backend and frontend together
├── run_haikyuu_chat.sh          # Optional Conda-aware launcher
├── requirements.txt             # Python dependencies
├── haikyuu_chat_ui/             # React/Vite frontend
├── haikyuu_db/                  # Source text files used for retrieval
└── haikyuu_images/              # UI/reference images
```

Generated folders such as `haikyuu_chroma/`, `haikyuu_chat_ui/node_modules/`, `haikyuu_chat_ui/dist/`, `__pycache__/`, and `unsloth_compiled_cache/` are intentionally ignored by Git.

## Requirements

- Python 3.10+
- Node.js and npm
- A Python environment that can install/run PyTorch and Unsloth
- CUDA-capable GPU recommended for the default model

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd haikyuu_chat_ui
npm install
cd ..
```

If you already use a Conda environment for Unsloth, activate it instead of creating `.venv`.

## Run

Start the API and UI together:

```bash
./run_haikyuu_chat.sh
```

Or from an already activated Python environment:

```bash
python run_haikyuu_chat.py
```

Open the UI at:

```text
http://127.0.0.1:5173/
```

The backend runs at `http://127.0.0.1:8001`.

## API

- `GET /health` - check backend and index status
- `POST /reindex` - rebuild the Chroma index from `haikyuu_db`
- `POST /retrieve` - inspect retrieved context for a question
- `POST /ask` - ask a non-streaming question
- `POST /ask/stream` - stream answer tokens as server-sent events

Example:

```bash
curl -X POST http://127.0.0.1:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Hinata like as a player?"}'
```

## Git Notes

Before the first push:

```bash
git init
git add .
git status
git commit -m "Initial commit"
```

Review `git status` before committing to confirm that generated model, cache, vector database, and dependency folders are not included.
