import unsloth
from pathlib import Path
import re
from threading import Lock

import chromadb
import torch
import yaml
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from unsloth import FastLanguageModel


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

source_folder = Path("haikyuu_db")
chroma_folder = Path("haikyuu_chroma")
collection_name = "haikyuu_txt_chunks"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
chunk_size = 1200
chunk_overlap = 200
max_context_chars = 6000
stop_words = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for",
    "from", "has", "have", "how", "i", "in", "is", "it", "of", "on", "or", "that",
    "the", "their", "them", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with",
}

source_folder.mkdir(exist_ok = True)
chroma_folder.mkdir(exist_ok = True)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # or choose "unsloth/Llama-3.2-1B-Instruct"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(embedding_model_name)
chroma_client = chromadb.PersistentClient(path = str(chroma_folder))
collection = chroma_client.get_or_create_collection(name = collection_name)
generate_lock = Lock()
index_lock = Lock()

app = FastAPI(title = "Haikyuu RAG API")

HOME_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Haikyuu RAG Chat</title>
    <style>
        :root {
            color-scheme: light dark;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        body {
            margin: 0;
            min-height: 100vh;
            background: #f5f7f8;
            color: #17202a;
        }
        main {
            width: min(960px, calc(100% - 32px));
            margin: 0 auto;
            padding: 32px 0;
        }
        h1 {
            margin: 0 0 20px;
            font-size: 28px;
            font-weight: 750;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 650;
        }
        textarea, input {
            box-sizing: border-box;
            width: 100%;
            border: 1px solid #b8c2cc;
            border-radius: 6px;
            padding: 12px;
            font: inherit;
            background: #ffffff;
            color: inherit;
        }
        textarea {
            min-height: 132px;
            resize: vertical;
        }
        .row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr 1fr auto auto;
            gap: 12px;
            align-items: end;
            margin-top: 16px;
        }
        button {
            border: 0;
            border-radius: 6px;
            padding: 13px 18px;
            font: inherit;
            font-weight: 700;
            background: #d95d1f;
            color: white;
            cursor: pointer;
        }
        button.secondary {
            background: #2f6f7e;
        }
        button:disabled {
            cursor: wait;
            opacity: 0.7;
        }
        pre {
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            min-height: 120px;
            margin-top: 24px;
            padding: 16px;
            border: 1px solid #d4dce3;
            border-radius: 6px;
            background: #ffffff;
            line-height: 1.5;
        }
        #sources {
            margin-top: 12px;
            line-height: 1.6;
        }
        details {
            margin-top: 18px;
        }
        summary {
            cursor: pointer;
            font-weight: 700;
        }
        .chunk {
            min-height: auto;
            margin-top: 12px;
            font-size: 14px;
        }
        @media (max-width: 900px) {
            .row {
                grid-template-columns: 1fr;
            }
        }
        @media (prefers-color-scheme: dark) {
            body {
                background: #111827;
                color: #e5e7eb;
            }
            textarea, input, pre {
                border-color: #374151;
                background: #1f2937;
            }
        }
    </style>
</head>
<body>
    <main>
        <h1>Haikyuu RAG Chat</h1>
        <form id="ask-form">
            <label for="question">Question</label>
            <textarea id="question" name="question" required>Who is Hinata Shoyo?</textarea>
            <div class="row">
                <div>
                    <label for="max_new_tokens">Max tokens</label>
                    <input id="max_new_tokens" name="max_new_tokens" type="number" min="1" max="2048" value="512">
                </div>
                <div>
                    <label for="top_k">Top K</label>
                    <input id="top_k" name="top_k" type="number" min="1" max="20" value="6">
                </div>
                <div>
                    <label for="temperature">Temperature</label>
                    <input id="temperature" name="temperature" type="number" min="0" max="2" step="0.1" value="1.0">
                </div>
                <div>
                    <label for="min_p">Min P</label>
                    <input id="min_p" name="min_p" type="number" min="0" max="1" step="0.05" value="0.1">
                </div>
                <button id="submit-button" type="submit">Ask</button>
                <button class="secondary" id="reindex-button" type="button">Reindex</button>
            </div>
        </form>
        <pre id="answer">Ask a question to see the model response here.</pre>
        <div id="sources"></div>
        <details open>
            <summary>Retrieved chunks</summary>
            <div id="chunks"></div>
        </details>
    </main>
    <script>
        const form = document.getElementById("ask-form");
        const button = document.getElementById("submit-button");
        const reindexButton = document.getElementById("reindex-button");
        const answer = document.getElementById("answer");
        const sources = document.getElementById("sources");
        const chunks = document.getElementById("chunks");

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            button.disabled = true;
            answer.textContent = "Generating...";
            sources.textContent = "";
            chunks.textContent = "";

            const payload = {
                question: document.getElementById("question").value,
                max_new_tokens: Number(document.getElementById("max_new_tokens").value),
                top_k: Number(document.getElementById("top_k").value),
                temperature: Number(document.getElementById("temperature").value),
                min_p: Number(document.getElementById("min_p").value),
            };

            try {
                const response = await fetch("/ask", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify(payload),
                });
                const data = await response.json();
                if (!response.ok) {
                    answer.textContent = JSON.stringify(data, null, 2);
                    return;
                }
                answer.textContent = data.answer || "(empty response)";
                sources.textContent = data.sources.length
                    ? "Sources: " + data.sources.map((source) => `${source.title} (${source.file}, chunk ${source.chunk_id}, ${source.section})`).join("; ")
                    : "Sources: none";
                chunks.innerHTML = "";
                if (!data.sources.length) {
                    chunks.textContent = "No chunks retrieved.";
                } else {
                    for (const source of data.sources) {
                        const block = document.createElement("pre");
                        block.className = "chunk";
                        block.textContent = `${source.title} (${source.file}, chunk ${source.chunk_id}, score ${source.score})\\nSection: ${source.section}\\n\\n${source.text}`;
                        chunks.appendChild(block);
                    }
                }
            } catch (error) {
                answer.textContent = String(error);
            } finally {
                button.disabled = false;
            }
        });

        reindexButton.addEventListener("click", async () => {
            reindexButton.disabled = true;
            answer.textContent = "Reindexing TXT files...";
            sources.textContent = "";
            chunks.textContent = "";
            try {
                const response = await fetch("/reindex", {method: "POST"});
                const data = await response.json();
                answer.textContent = JSON.stringify(data, null, 2);
            } catch (error) {
                answer.textContent = String(error);
            } finally {
                reindexButton.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length = 1)
    max_new_tokens: int = Field(512, ge = 1, le = 2048)
    top_k: int = Field(6, ge = 1, le = 20)
    temperature: float = Field(1.0, ge = 0.0, le = 2.0)
    min_p: float = Field(0.1, ge = 0.0, le = 1.0)


class Source(BaseModel):
    file: str
    title: str
    chunk_id: int
    section: str = ""
    text: str = ""
    score: float = 0.0


class QuestionResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    max_new_tokens: int


class ReindexResponse(BaseModel):
    status: str
    files: int
    chunks: int
    source_folder: str
    chroma_folder: str


def source_file_name(path):
    return path.relative_to(source_folder).as_posix()


def parse_txt_file(path):
    text = path.read_text(encoding = "utf-8")
    relative_path = path.relative_to(source_folder)
    metadata = {
        "title": path.stem,
        "file": source_file_name(path),
        "folder": relative_path.parent.as_posix(),
        "category": path.stem.split("_")[-1],
    }
    if len(relative_path.parts) >= 3 and relative_path.parts[0] == "characters":
        metadata["character_slug"] = relative_path.parts[1]

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            parsed_metadata = yaml.safe_load(parts[1]) or {}
            for key, value in parsed_metadata.items():
                if value is None:
                    continue
                if isinstance(value, list):
                    metadata[key] = ", ".join(str(item) for item in value)
                else:
                    metadata[key] = str(value)
            text = parts[2].strip()

    return metadata, text.strip()


def metadata_search_text(metadata):
    fields = [
        metadata.get("title", ""),
        metadata.get("type", ""),
        metadata.get("subject", ""),
        metadata.get("canonical_name", ""),
        metadata.get("aliases", ""),
        metadata.get("source_scope", ""),
        metadata.get("characters", ""),
        metadata.get("teams", ""),
        metadata.get("topics", ""),
        metadata.get("season", ""),
        metadata.get("section", ""),
        metadata.get("file", ""),
        metadata.get("folder", ""),
        metadata.get("character_slug", ""),
        metadata.get("category", ""),
    ]
    return "\n".join(str(field) for field in fields if field)


def infer_section(chunk, fallback):
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= 120:
            return line
        break
    return fallback


def chunk_text(text):
    units = [unit.strip() for unit in re.split(r"\n\s*\n", text) if unit.strip()]
    if len(units) <= 1:
        units = [unit.strip() for unit in text.splitlines() if unit.strip()]

    chunks = []
    current = ""

    for unit in units:
        if len(unit) > chunk_size:
            if current:
                chunks.append(current.strip())
                current = ""
            start = 0
            while start < len(unit):
                end = min(start + chunk_size, len(unit))
                chunk = unit[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                if end == len(unit):
                    break
                start = max(end - chunk_overlap, start + 1)
            continue

        candidate = f"{current}\n\n{unit}".strip() if current else unit
        if len(candidate) > chunk_size and current:
            chunks.append(current.strip())
            current = unit
        else:
            current = candidate

    if current:
        chunks.append(current.strip())

    return chunks


def embed_texts(texts):
    embeddings = embedding_model.encode(texts, normalize_embeddings = True)
    return embeddings.tolist()


def search_tokens(text):
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in stop_words}


def keyword_score(question, document, metadata):
    query_tokens = search_tokens(question)
    if not query_tokens:
        return 0
    haystack = f"{metadata_search_text(metadata)}\n{document}"
    haystack_tokens = search_tokens(haystack)
    return len(query_tokens & haystack_tokens)


def get_collection():
    global collection
    collection = chroma_client.get_or_create_collection(name = collection_name)
    return collection


def txt_file_count():
    count = 0
    for path in source_folder.rglob("*.txt"):
        _, text = parse_txt_file(path)
        if text:
            count += 1
    return count


def rebuild_index():
    global collection
    with index_lock:
        try:
            chroma_client.delete_collection(name = collection_name)
        except Exception:
            pass
        collection = chroma_client.get_or_create_collection(name = collection_name)

        ids = []
        documents = []
        metadatas = []
        file_count = 0

        for path in sorted(source_folder.rglob("*.txt")):
            metadata, text = parse_txt_file(path)
            if not text:
                continue
            file_count += 1
            for chunk_id, chunk in enumerate(chunk_text(text), start = 1):
                chunk_metadata = dict(metadata)
                chunk_metadata["chunk_id"] = chunk_id
                chunk_metadata["file"] = source_file_name(path)
                chunk_metadata["title"] = chunk_metadata.get("title", path.stem)
                chunk_metadata["section"] = infer_section(chunk, chunk_metadata["title"])

                indexed_chunk = f"{metadata_search_text(chunk_metadata)}\n\n{chunk}"

                ids.append(f"{source_file_name(path)}:{chunk_id}")
                documents.append(indexed_chunk)
                metadatas.append(chunk_metadata)

        if documents:
            collection.add(
                ids = ids,
                documents = documents,
                embeddings = embed_texts(documents),
                metadatas = metadatas,
            )

        return {"files": file_count, "chunks": len(documents)}


def retrieve_context(question, top_k):
    active_collection = get_collection()
    if active_collection.count() == 0:
        return "", []

    candidate_count = min(active_collection.count(), max(top_k * 4, 16))
    results = active_collection.query(
        query_embeddings = embed_texts([question]),
        n_results = candidate_count,
        include = ["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    candidates = []
    seen = set()

    for index, (document, metadata) in enumerate(zip(documents, metadatas)):
        identity = (metadata.get("file", ""), metadata.get("chunk_id", 0))
        seen.add(identity)
        distance = distances[index] if index < len(distances) else 1.0
        lexical_score = keyword_score(question, document, metadata)
        semantic_score = 1.0 / (1.0 + float(distance))
        combined_score = semantic_score + (lexical_score * 0.18)
        candidates.append((combined_score, document, metadata))

    all_results = active_collection.get(include = ["documents", "metadatas"])
    all_documents = all_results.get("documents", [])
    all_metadatas = all_results.get("metadatas", [])
    lexical_candidates = []

    for document, metadata in zip(all_documents, all_metadatas):
        identity = (metadata.get("file", ""), metadata.get("chunk_id", 0))
        if identity in seen:
            continue
        lexical_score = keyword_score(question, document, metadata)
        if lexical_score > 0:
            lexical_candidates.append((lexical_score * 0.18, document, metadata))

    lexical_candidates.sort(key = lambda item: item[0], reverse = True)
    candidates.extend(lexical_candidates[:candidate_count])
    candidates.sort(key = lambda item: item[0], reverse = True)

    context_blocks = []
    sources = []
    current_chars = 0

    for combined_score, document, metadata in candidates[:top_k]:
        source = Source(
            file = metadata.get("file", ""),
            title = metadata.get("title", metadata.get("file", "")),
            chunk_id = int(metadata.get("chunk_id", 0)),
            section = metadata.get("section", ""),
            text = document,
            score = round(combined_score, 4),
        )
        block = f"[Source: {source.title} | File: {source.file} | Chunk: {source.chunk_id} | Section: {source.section}]\n{document}"
        if current_chars + len(block) > max_context_chars:
            break
        context_blocks.append(block)
        sources.append(source)
        current_chars += len(block)

    return "\n\n".join(context_blocks), sources


def build_rag_prompt(question, context):
    return (
        "Use the Haikyuu database context below to answer the user's question.\n"
        "The context may contain episode summaries, character notes, statistics, timeline entries, or match summaries.\n"
        "If a retrieved chunk is relevant, answer from it even when the wording does not exactly match the question.\n"
        "If the context does not contain enough information, say: "
        "\"The current Haikyuu database does not contain enough information to answer that.\" "
        "Do not invent facts that are not supported by the context.\n\n"
        f"Haikyuu database context:\n{context}\n\n"
        f"Question: {question}"
    )


def answer_question(question, max_new_tokens = 512, top_k = 6, temperature = 1.0, min_p = 0.1):
    context, sources = retrieve_context(question, top_k)
    if context == "":
        return (
            "The current Haikyuu database does not contain enough information to answer that.",
            sources,
        )

    messages = [
        {"role": "system", "content": "You are a Haikyuu RAG assistant. Answer using only the provided database context."},
        {"role": "user", "content": build_rag_prompt(question, context)},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
    )
    inputs = tokenizer(text, return_tensors = "pt").to(device)

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    if temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["min_p"] = min_p

    with generate_lock:
        with torch.inference_mode():
            outputs = model.generate(**generation_kwargs)

    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens = True)
    return answer.strip(), sources


@app.get("/", response_class = HTMLResponse)
def root():
    return HOME_PAGE


@app.get("/favicon.ico", include_in_schema = False)
def favicon():
    return Response(status_code = 204)


@app.get("/health")
def health():
    active_collection = get_collection()
    return {
        "status": "ok",
        "device": device,
        "source_folder": str(source_folder),
        "chroma_folder": str(chroma_folder),
        "collection": collection_name,
        "chunks": active_collection.count(),
        "txt_files_with_content": txt_file_count(),
        "embedding_model": embedding_model_name,
    }


@app.post("/reindex", response_model = ReindexResponse)
def reindex():
    stats = rebuild_index()
    return ReindexResponse(
        status = "ok",
        files = stats["files"],
        chunks = stats["chunks"],
        source_folder = str(source_folder),
        chroma_folder = str(chroma_folder),
    )


@app.post("/retrieve")
def retrieve(request: QuestionRequest):
    context, sources = retrieve_context(request.question, request.top_k)
    return {
        "question": request.question,
        "sources": sources,
        "context": context,
    }


@app.post("/ask", response_model = QuestionResponse)
def ask(request: QuestionRequest):
    answer, sources = answer_question(
        request.question,
        max_new_tokens = request.max_new_tokens,
        top_k = request.top_k,
        temperature = request.temperature,
        min_p = request.min_p,
    )
    return QuestionResponse(
        question = request.question,
        answer = answer,
        sources = sources,
        max_new_tokens = request.max_new_tokens,
    )
