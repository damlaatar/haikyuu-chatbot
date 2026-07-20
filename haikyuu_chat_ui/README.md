# Haikyuu Chat UI

React/Vite frontend for `haikyuu_rag_chat_api.py`.

## Run

From the project root, run both the streaming RAG API and React UI:

```bash
cd /home/serdoriko/projects/chatbot
./run_haikyuu_chat.sh
```

The launcher starts:

- FastAPI backend: `http://127.0.0.1:8001`
- React UI: `http://127.0.0.1:5173`

It also runs `npm install` automatically if `haikyuu_chat_ui/node_modules` is missing.

If you already activated `unsloth_1`, this also works:

```bash
python run_haikyuu_chat.py
```

Open:

```text
http://127.0.0.1:5173/
```

## Optional API URL

If the API runs somewhere else:

```bash
VITE_HAIKYUU_API_URL=http://127.0.0.1:8001 npm run dev
```
