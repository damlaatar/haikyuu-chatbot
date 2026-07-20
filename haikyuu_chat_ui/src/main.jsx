import React, { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_HAIKYUU_API_URL || "http://127.0.0.1:8001";

const initialMessages = [
  {
    id: "welcome",
    role: "assistant",
    text: "Ask me anything about Haikyuu. I can help with characters, matches, quotes, stats, and story details.",
    sources: [],
    done: true,
  },
];

function SourceList({ sources, enabled }) {
  if (!enabled || !sources || sources.length === 0) {
    return null;
  }

  return (
    <details className="sources">
      <summary>Retrieved chunks</summary>
      <div className="source-grid">
        {sources.map((source, index) => (
          <article className="source-card" key={`${source.file}-${source.chunk_id}-${index}`}>
            <div className="source-topline">
              <strong>{source.title || source.file}</strong>
              <span>score {source.score}</span>
            </div>
            <p>{source.file} · chunk {source.chunk_id}</p>
            {source.section ? <p className="section">{source.section}</p> : null}
            <pre>{source.text}</pre>
          </article>
        ))}
      </div>
    </details>
  );
}

function MessageBubble({ message, showSources }) {
  const label = message.role === "user" ? "You" : "AI";

  return (
    <div className={`message-row ${message.role}`}>
      <div className="message-avatar" aria-hidden="true">{label}</div>
      <div className="message-bubble">
        <div className="message-label">{message.role === "user" ? "You" : "Haikyuu DB"}</div>
        <p>{message.text || (message.done ? "" : "Thinking...")}</p>
        {!message.done && <div className="typing-bar" aria-label="Generating answer" />}
        <SourceList sources={message.sources} enabled={showSources} />
      </div>
    </div>
  );
}

function NumberControl({ label, value, min, max, step = 1, onChange }) {
  return (
    <label className="control">
      <span>{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function ToggleControl({ label, checked, onChange }) {
  return (
    <label className="toggle-control">
      <span>{label}</span>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
      />
    </label>
  );
}

function parseStreamEvent(buffer) {
  const events = [];
  let remaining = buffer;
  let boundary = remaining.indexOf("\n\n");

  while (boundary !== -1) {
    const rawEvent = remaining.slice(0, boundary);
    remaining = remaining.slice(boundary + 2);
    const dataLine = rawEvent.split("\n").find((line) => line.startsWith("data: "));
    if (dataLine) {
      try {
        events.push(JSON.parse(dataLine.slice(6)));
      } catch (error) {
        events.push({ type: "error", message: String(error) });
      }
    }
    boundary = remaining.indexOf("\n\n");
  }

  return { events, remaining };
}

function App() {
  const [messages, setMessages] = useState(initialMessages);
  const [question, setQuestion] = useState("");
  const [settings, setSettings] = useState({
    max_new_tokens: 768,
    top_k: 8,
    temperature: 0.7,
    min_p: 0.1,
  });
  const [showSources, setShowSources] = useState(false);
  const [status, setStatus] = useState("Ready");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isReindexing, setIsReindexing] = useState(false);
  const chatRef = useRef(null);

  function updateSetting(key, value) {
    setSettings((current) => ({ ...current, [key]: value }));
  }

  function scrollToBottom() {
    window.requestAnimationFrame(() => {
      chatRef.current?.scrollTo({ top: chatRef.current.scrollHeight, behavior: "smooth" });
    });
  }

  async function submitQuestion(event) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isGenerating) {
      return;
    }

    const userMessage = {
      id: crypto.randomUUID(),
      role: "user",
      text: trimmedQuestion,
      sources: [],
      done: true,
    };
    const assistantId = crypto.randomUUID();
    const assistantMessage = {
      id: assistantId,
      role: "assistant",
      text: "",
      sources: [],
      done: false,
    };

    setMessages((current) => [...current, userMessage, assistantMessage]);
    setQuestion("");
    setStatus("Generating answer...");
    setIsGenerating(true);
    scrollToBottom();

    try {
      const response = await fetch(`${API_BASE_URL}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmedQuestion,
          ...settings,
        }),
      });

      if (!response.ok || !response.body) {
        const errorText = await response.text();
        throw new Error(errorText || `Request failed with ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const parsed = parseStreamEvent(buffer);
        buffer = parsed.remaining;

        for (const eventData of parsed.events) {
          if (eventData.type === "token") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, text: message.text + eventData.text }
                  : message,
              ),
            );
          }

          if (eventData.type === "done") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, sources: eventData.sources || [], done: true }
                  : message,
              ),
            );
          }
        }
        scrollToBottom();
      }

      setStatus("Ready");
    } catch (error) {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? { ...message, text: `Request failed: ${String(error.message || error)}`, done: true }
            : message,
        ),
      );
      setStatus("Request failed");
    } finally {
      setIsGenerating(false);
      scrollToBottom();
    }
  }

  async function reindexDatabase() {
    if (isReindexing) {
      return;
    }

    setIsReindexing(true);
    setStatus("Reindexing database...");

    try {
      const response = await fetch(`${API_BASE_URL}/reindex`, { method: "POST" });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(JSON.stringify(data));
      }
      setStatus(`Indexed ${data.files} files into ${data.chunks} chunks`);
    } catch (error) {
      setStatus(`Reindex failed: ${String(error.message || error)}`);
    } finally {
      setIsReindexing(false);
    }
  }

  function handleComposerKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitQuestion(event);
    }
  }

  return (
    <main className="app-shell">
      <aside className="settings-sidebar" aria-label="Chat settings">
        <div className="brand-block">
          <p className="eyebrow">Local Haikyuu RAG</p>
          <h1>Karasuno Club Room</h1>
          <div className="status-pill">{status}</div>
        </div>

        <section className="settings-card" aria-label="Model controls">
          <h2>Model Settings</h2>
          <NumberControl
            label="Max tokens"
            min={1}
            max={2048}
            value={settings.max_new_tokens}
            onChange={(value) => updateSetting("max_new_tokens", value)}
          />
          <NumberControl
            label="Top K"
            min={1}
            max={20}
            value={settings.top_k}
            onChange={(value) => updateSetting("top_k", value)}
          />
          <NumberControl
            label="Temperature"
            min={0}
            max={2}
            step={0.1}
            value={settings.temperature}
            onChange={(value) => updateSetting("temperature", value)}
          />
          <NumberControl
            label="Min P"
            min={0}
            max={1}
            step={0.05}
            value={settings.min_p}
            onChange={(value) => updateSetting("min_p", value)}
          />
          <ToggleControl label="Show retrieved chunks" checked={showSources} onChange={setShowSources} />
        </section>

        <section className="settings-actions" aria-label="Database actions">
          <button className="reindex-button" type="button" disabled={isReindexing} onClick={reindexDatabase}>
            {isReindexing ? "Indexing" : "Reindex Database"}
          </button>
        </section>
      </aside>

      <section className="chat-shell">
        <section className="chat-window" ref={chatRef} aria-label="Chat messages">
          {messages.map((message) => (
            <MessageBubble message={message} showSources={showSources} key={message.id} />
          ))}
        </section>

        <form className="composer" onSubmit={submitQuestion}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleComposerKeyDown}
            placeholder="Ask about matches, characters, quotes, stats, or relationships..."
            rows={2}
          />
          <button type="submit" disabled={isGenerating || !question.trim()}>
            {isGenerating ? "Generating" : "Send"}
          </button>
        </form>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
