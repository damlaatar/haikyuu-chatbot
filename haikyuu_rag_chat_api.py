import json
import re
from pathlib import Path
from threading import Lock, Thread

import chromadb
import torch
import unsloth
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from transformers import TextIteratorStreamer
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

query_aliases = {
    "aoba johsai": "seijoh oikawa iwaizumi",
    "seijoh": "aoba johsai oikawa iwaizumi",
    "shiratorizawa": "ushijima tendo goshiki",
    "nekoma": "kenma kuroo battle at the garbage dump",
    "fukurodani": "bokuto akaashi",
    "karasuno": "hinata kageyama daichi sugawara tanaka nishinoya asahi tsukishima yamaguchi",
    "miya": "atsumu osamu inarizaki",
    "match": "game set score tournament arc episode",
    "vs": "match game set score tournament arc episode",
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

app = FastAPI(title = "Haikyuu RAG Chat API")
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length = 1)
    max_new_tokens: int = Field(768, ge = 1, le = 2048)
    top_k: int = Field(8, ge = 1, le = 20)
    temperature: float = Field(0.7, ge = 0.0, le = 2.0)
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


def expand_query(question):
    expanded_terms = []
    lowered = question.lower()
    for key, value in query_aliases.items():
        if key in lowered:
            expanded_terms.append(value)

    useful_tokens = sorted(search_tokens(question))
    if useful_tokens:
        expanded_terms.append(" ".join(useful_tokens))

    if not expanded_terms:
        return question

    return f"{question}\nSearch keywords: {' '.join(expanded_terms)}"


def keyword_score(question, document, metadata):
    query_tokens = search_tokens(expand_query(question))
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

    retrieval_query = expand_query(question)
    candidate_count = min(active_collection.count(), max(top_k * 5, 24))
    results = active_collection.query(
        query_embeddings = embed_texts([retrieval_query]),
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
        combined_score = semantic_score + (lexical_score * 0.22)
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
            lexical_candidates.append((lexical_score * 0.22, document, metadata))

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
        "Answer the user's Haikyuu question using the database context below as the factual authority.\n"
        "Answer in a normal conversational style and match the length to the question.\n"
        "For simple factual questions, give the answer in 1 or 2 concise sentences.\n"
        "For explanation, comparison, relationship, or story questions, give a fuller answer with the important details.\n"
        "Do not pad simple answers with meta-commentary, caveats, or statements about what the series does not focus on unless the user asks for that.\n"
        "Start with the direct answer, then add only useful context.\n"
        "Do not include a separate evidence, source, citation, or retrieved-chunk section in the answer. "
        "Do not say phrases like \"provided database context\", \"according to the context\", or \"mentioned in the context\" unless the user asks for sources.\n"
        "Use the retrieved context silently to decide what is true, and keep the chat response readable.\n"
        "If the context partially answers the question, answer the supported part and clearly say what is missing.\n"
        "If the retrieved context is unrelated or empty, say: "
        "\"The current Haikyuu database does not contain enough information to answer that.\" "
        "You may use general Haikyuu knowledge only to understand the question and write naturally; "
        "do not introduce unsupported facts as if they came from the database.\n\n"
        f"Haikyuu database context:\n{context}\n\n"
        f"Question: {question}"
    )


def build_messages(question, context):
    return [
        {
            "role": "system",
            "content": (
                "You are a warm, decisive Haikyuu RAG assistant. "
                "The database context is authoritative. Focus on the user's actual question. "
                "Be concise for simple facts and more detailed only when the question needs explanation. "
                "Do not expose your evidence unless asked."
            ),
        },
        {"role": "user", "content": build_rag_prompt(question, context)},
    ]


def build_generation_kwargs(question, context, max_new_tokens, temperature, min_p, streamer = None):
    text = tokenizer.apply_chat_template(
        build_messages(question, context),
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

    if streamer is not None:
        generation_kwargs["streamer"] = streamer

    if temperature <= 0:
        generation_kwargs["do_sample"] = False
    else:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
        generation_kwargs["min_p"] = min_p

    return inputs, generation_kwargs


def answer_question(question, max_new_tokens = 512, top_k = 6, temperature = 0.7, min_p = 0.1):
    context, sources = retrieve_context(question, top_k)
    if context == "":
        return (
            "The current Haikyuu database does not contain enough information to answer that.",
            sources,
        )

    inputs, generation_kwargs = build_generation_kwargs(
        question,
        context,
        max_new_tokens,
        temperature,
        min_p,
    )

    with generate_lock:
        with torch.inference_mode():
            outputs = model.generate(**generation_kwargs)

    answer = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens = True)
    return answer.strip(), sources


def sse_event(event_type, payload):
    data = {"type": event_type, **payload}
    return f"data: {json.dumps(data, ensure_ascii = False)}\n\n"


def source_to_dict(source):
    if hasattr(source, "model_dump"):
        return source.model_dump()
    return source.dict()


def run_streamed_generation(streamer, generation_kwargs):
    try:
        with torch.inference_mode():
            model.generate(**generation_kwargs)
    except Exception as error:
        streamer.on_finalized_text(f"\nGeneration failed: {error}", stream_end = True)


def stream_answer_events(request):
    context, sources = retrieve_context(request.question, request.top_k)
    if context == "":
        yield sse_event("token", {"text": "The current Haikyuu database does not contain enough information to answer that."})
        yield sse_event("done", {"sources": [source_to_dict(source) for source in sources]})
        return

    streamer = TextIteratorStreamer(tokenizer, skip_prompt = True, skip_special_tokens = True)
    _, generation_kwargs = build_generation_kwargs(
        request.question,
        context,
        request.max_new_tokens,
        request.temperature,
        request.min_p,
        streamer = streamer,
    )

    with generate_lock:
        thread = Thread(target = run_streamed_generation, args = (streamer, generation_kwargs))
        thread.start()
        for token in streamer:
            if token:
                yield sse_event("token", {"text": token})
        thread.join()

    yield sse_event("done", {"sources": [source_to_dict(source) for source in sources]})


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Run the React app in haikyuu_chat_ui and call this API from http://127.0.0.1:5173.",
    }


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
        "expanded_query": expand_query(request.question),
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


@app.post("/ask/stream")
def ask_stream(request: QuestionRequest):
    return StreamingResponse(
        stream_answer_events(request),
        media_type = "text/event-stream",
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
