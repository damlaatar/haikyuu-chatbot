from threading import Lock

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from unsloth import FastLanguageModel


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

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
generate_lock = Lock()

app = FastAPI(title = "Llama 3.2 Inference API")

HOME_PAGE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Llama 3.2 Chat</title>
    <style>
        :root {
            color-scheme: light dark;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        body {
            margin: 0;
            min-height: 100vh;
            background: #f4f6f8;
            color: #17202a;
        }
        main {
            width: min(900px, calc(100% - 32px));
            margin: 0 auto;
            padding: 32px 0;
        }
        h1 {
            margin: 0 0 20px;
            font-size: 28px;
            font-weight: 700;
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
            min-height: 130px;
            resize: vertical;
        }
        .row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr auto;
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
            background: #1f7a5b;
            color: white;
            cursor: pointer;
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
        @media (max-width: 760px) {
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
        <h1>Llama 3.2 Chat</h1>
        <form id="ask-form">
            <label for="question">Question</label>
            <textarea id="question" name="question" required>What is the difference between machine learning and deep learning?</textarea>
            <div class="row">
                <div>
                    <label for="max_new_tokens">Max tokens</label>
                    <input id="max_new_tokens" name="max_new_tokens" type="number" min="1" max="2048" value="512">
                </div>
                <div>
                    <label for="temperature">Temperature</label>
                    <input id="temperature" name="temperature" type="number" min="0" max="2" step="0.1" value="1.5">
                </div>
                <div>
                    <label for="min_p">Min P</label>
                    <input id="min_p" name="min_p" type="number" min="0" max="1" step="0.05" value="0.1">
                </div>
                <button id="submit-button" type="submit">Ask</button>
            </div>
        </form>
        <pre id="answer">Ask a question to see the model response here.</pre>
    </main>
    <script>
        const form = document.getElementById("ask-form");
        const button = document.getElementById("submit-button");
        const answer = document.getElementById("answer");

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            button.disabled = true;
            answer.textContent = "Generating...";

            const payload = {
                question: document.getElementById("question").value,
                max_new_tokens: Number(document.getElementById("max_new_tokens").value),
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
            } catch (error) {
                answer.textContent = String(error);
            } finally {
                button.disabled = false;
            }
        });
    </script>
</body>
</html>
"""


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length = 1)
    max_new_tokens: int = Field(512, ge = 1, le = 2048)
    temperature: float = Field(1.5, ge = 0.0, le = 2.0)
    min_p: float = Field(0.1, ge = 0.0, le = 1.0)


class QuestionResponse(BaseModel):
    question: str
    answer: str
    max_new_tokens: int


def answer_question(question, max_new_tokens = 512, temperature = 1.5, min_p = 0.1):
    messages = [
        {"role": "user", "content": question},
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
    return answer.strip()


@app.get("/", response_class = HTMLResponse)
def root():
    return HOME_PAGE


@app.get("/favicon.ico", include_in_schema = False)
def favicon():
    return Response(status_code = 204)


@app.get("/health")
def health():
    return {"status": "ok", "device": device}


@app.post("/ask", response_model = QuestionResponse)
def ask(request: QuestionRequest):
    answer = answer_question(
        request.question,
        max_new_tokens = request.max_new_tokens,
        temperature = request.temperature,
        min_p = request.min_p,
    )
    return QuestionResponse(
        question = request.question,
        answer = answer,
        max_new_tokens = request.max_new_tokens,
    )
