const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const button = document.getElementById("submit");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const answerPanel = document.getElementById("answer-panel");
const answerEl = document.getElementById("answer");
const metaEl = document.getElementById("meta");
const sourcesPanel = document.getElementById("sources-panel");
const sourcesEl = document.getElementById("sources");
const sourcesCount = document.getElementById("sources-count");

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = "status status--" + kind;
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = !message;
}

async function checkHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();

    if (!data.index_ready) {
      setStatus("Index not loaded — " + (data.index_error || "unknown error"), "err");
      return;
    }
    const chunks = data.index.chunks;
    const files = data.index.sources.length;
    if (data.llm && data.llm.model) {
      setStatus(`${chunks} chunks from ${files} file(s) · model ${data.llm.model}`, "ok");
    } else {
      setStatus(`${chunks} chunks from ${files} file(s) · Ollama unavailable`, "warn");
    }
  } catch (err) {
    setStatus("Server unreachable: " + err.message, "err");
  }
}

function renderSources(sources) {
  sourcesEl.innerHTML = "";
  if (!sources || sources.length === 0) {
    sourcesPanel.hidden = true;
    return;
  }
  sourcesCount.textContent = `(${sources.length})`;
  for (const s of sources) {
    const li = document.createElement("li");

    const details = document.createElement("details");
    const summary = document.createElement("summary");

    const head = document.createElement("span");
    head.className = "src-head";
    head.textContent = `${s.source} · chunk ${s.chunk} · similarity ${s.score.toFixed(3)}`;

    const preview = document.createElement("span");
    preview.className = "src-preview";
    preview.textContent = s.text.length > 160 ? s.text.slice(0, 160) + "…" : s.text;

    const body = document.createElement("div");
    body.className = "src-text";
    body.textContent = s.text;

    summary.append(head, preview);
    details.append(summary, body);
    li.appendChild(details);
    sourcesEl.appendChild(li);
  }
  sourcesPanel.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  button.disabled = true;
  showError("");
  sourcesPanel.hidden = true;
  answerPanel.hidden = false;
  answerEl.innerHTML = '<span class="dots">Thinking</span>';
  metaEl.textContent = "";

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    if (data.answer) {
      answerEl.textContent = data.answer;
      metaEl.textContent = data.model ? `Answered by ${data.model}` : "";
    } else {
      answerPanel.hidden = true;
      showError(data.error || "No answer returned.");
    }
    renderSources(data.sources);
  } catch (err) {
    answerPanel.hidden = true;
    showError("Request failed: " + err.message);
  } finally {
    button.disabled = false;
    input.focus();
  }
});

checkHealth();
input.focus();
