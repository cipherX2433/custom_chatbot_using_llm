<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/Academic%20Assistant-CSV%20RAG-0b3d91?style=for-the-badge&logo=github&logoColor=white">
    <img alt="Academic Assistant (CSV RAG)" src="https://img.shields.io/badge/Academic%20Assistant-CSV%20RAG-1f6feb?style=for-the-badge&logo=github">
  </picture>
</p>

<h1 align="center">📚 Academic Assistant (CSV‑based RAG)</h1>

<p align="center">
  A Retrieval‑Augmented Generation (RAG) assistant for VIT University that answers student queries using
  <b>CSV</b> data → <b>Chroma VectorDB</b> → <b>Ollama (Llama2)</b> with <b>HuggingFace Embeddings</b>.
</p>

<p align="center">
  <a href="#-features"><img alt="Features" src="https://img.shields.io/badge/Features-9-10b981?style=flat-square"></a>
  <a href="#%EF%B8%8F-tech-stack"><img alt="Tech" src="https://img.shields.io/badge/Tech-Stack-6366f1?style=flat-square"></a>
  <a href="#%EF%B8%8F-installation"><img alt="Install" src="https://img.shields.io/badge/Install-Guide-22c55e?style=flat-square"></a>
  <a href="#-usage"><img alt="Usage" src="https://img.shields.io/badge/Usage-CLI-06b6d4?style=flat-square"></a>
  <a href="#-project-structure"><img alt="Tree" src="https://img.shields.io/badge/Project-Tree-f59e0b?style=flat-square"></a>
  <a href="#-contributing"><img alt="Contrib" src="https://img.shields.io/badge/PRs-Welcome-f472b6?style=flat-square"></a>
  <a href="#-license"><img alt="License" src="https://img.shields.io/badge/License-MIT-111827?style=flat-square"></a>
</p>

---

## ✨ Overview

This project implements a **CSV‑powered RAG assistant** for academic data (courses, subjects, credits, prerequisites,
labs, electives, etc.). It uses **Chroma** as a vector store, **Alibaba-NLP gte-large-en-v1.5** embeddings for semantic
search, and **Ollama (Llama2)** to generate structured, student‑friendly answers. The assistant avoids hallucinations
and will reply with **“I don’t know”** when data is missing.

> Bonus: Optional **AGNO Agent** integration with **Gemini + Reasoning Tools** to orchestrate complex tool use.

## 🚀 Features

* ✅ Load academic data from a **CSV** into **Chroma VectorDB**
* ✅ Use **Alibaba‑NLP `gte-large-en-v1.5`** embeddings for semantic search
* ✅ Retrieve chunks with **`RecursiveCharacterTextSplitter`** (LangChain)
* ✅ Generate answers with **Ollama (Llama2 local LLM)**
* ✅ Accuracy‑first: gracefully answers **“I don’t know”** if the data is missing
* ✅ Outputs **tables** and neat formatting when possible
* ✅ Optional **AGNO Agent** with **Gemini + Reasoning Tools**

## 🧠 RAG Flow (Mermaid)

```mermaid
flowchart LR
  A[CSV Dataset\nresult.csv] -->|Load & Clean| B[Chunking\nRecursiveCharacterTextSplitter]
  B --> C[Embedding\nAlibaba-NLP gte-large-en-v1.5]
  C --> D[Chroma VectorDB]
  E[User Query] --> F[Retriever]
  F --> D
  D -->|Top‑K Chunks| G[LLM: Ollama (Llama2)]
  G --> H[Structured Answer\n(tables, citations, \"I don't know\")]
  subgraph Optional Agent Layer
    I[AGNO Agent] --> F
    I --> G
  end
```

## 🛠️ Tech Stack

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-Tooling-1a7f37?logo=chainlink&logoColor=white">
  <img alt="Chroma" src="https://img.shields.io/badge/Chroma-VectorDB-8b5cf6">
  <img alt="HuggingFace" src="https://img.shields.io/badge/HuggingFace-Embeddings-ffcc00?logo=huggingface&logoColor=black">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-Llama2-0ea5e9">
  <img alt="dotenv" src="https://img.shields.io/badge/dotenv-Config-4ade80">
  <img alt="AGNO" src="https://img.shields.io/badge/AGNO-Optional-f97316">
</p>

## 📂 Project Structure

```
.
├── resources/
│   └── academics/         # Chroma vectorstore directory
├── result.csv             # CSV dataset with academic information
├── academic_assistant.py  # Main script
└── README.md              # Documentation
```

## ⚙️ Installation

> Replace `yourusername/academic-assistant` with your GitHub handle and repository name.

```bash
# 1) Clone the repository
git clone https://github.com/yourusername/academic-assistant.git
cd academic-assistant

# 2) Create & activate a virtual environment
python3 -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt

# 4) Install Ollama (for local Llama2)
#   – macOS:  https://ollama.com/download
#   – Linux:  curl -fsSL https://ollama.com/install.sh | sh
#   – Windows: https://ollama.com/download

# 5) Pull the Llama2 model
ollama pull llama2
```

### 🔐 Environment Variables

Create a `.env` file in the project root:

```dotenv
# Avoids tokenizer parallelism warnings
TOKENIZERS_PARALLELISM=false
```

> If you add keys for optional services (e.g., Gemini for AGNO Agent), document them here too.

## ▶️ Usage

```bash
# Index the CSV and start answering queries
python academic_assistant.py --csv ./result.csv --persist ./resources/academics

# Example flags (customize as needed)
# --k 4                         # top-k chunks to retrieve
# --model llama2                # Ollama model name
# --embedding gte-large-en-v1.5 # embedding model id
# --max-tokens 512              # generation cap
```

<details>
  <summary><b>Sample Query</b></summary>

```text
User: What are the prerequisites for CSE2004 and how many credits is it?
Assistant:
- Course: CSE2004 — Data Structures
- Credits: 4
- Prerequisites: CSE1001 or equivalent foundation (per CSV row #42)
```

</details>

## 📊 Output Format

* Answers are **fact‑grounded** in the CSV (citation‑style references to row/field when possible)
* Uses **tables** for course lists and **bullet points** for requirements
* If information is not in the dataset, responds with **“I don’t know”**

> Example Table (rendered by the assistant):

| Course Code | Title            | Credits | Prerequisites   |
| ----------- | ---------------- | ------- | --------------- |
| CSE2004     | Data Structures  | 4       | CSE1001         |
| CSE3001     | Machine Learning | 3       | Probability, DS |

## 🧪 Development Notes

* The first run will **create/persist** a Chroma DB under `./resources/academics/`
* Re‑index if `result.csv` changes (or implement file hashing to auto‑refresh)
* Prefer **deterministic** prompts and **top‑k** tuning for reliability

## 🧩 Optional: AGNO Agent + Gemini

If you enable the AGNO Agent orchestration layer, you can:

* Route queries through **tools** (e.g., tabular filtering, CSV aggregation)
* Use **Gemini + Reasoning Tools** for complex planning
* Keep Ollama/Llama2 as the final answer generator

> Add any required keys to `.env` and document your agent config under `./agents/`.

## 🐛 Troubleshooting

* **Ollama not found** → install from the official site and ensure the daemon is running
* **Model missing** → run `ollama pull llama2`
* **Tokenizer warnings** → ensure `TOKENIZERS_PARALLELISM=false` in `.env`
* **Slow/empty retrievals** → check CSV cleanliness, chunk size/overlap, and embedding model id

## 🤝 Contributing

PRs welcome! Please:

1. Fork the repo
2. Create a feature branch (`feat/my-awesome-thing`)
3. Commit with conventional messages
4. Open a PR with screenshots or terminal output

## 🧭 Roadmap

* [ ] Add evaluation suite (answer faithfulness & coverage)
* [ ] Web UI (FastAPI + React) for student queries
* [ ] Better table rendering + column alignment
* [ ] Multi‑CSV ingestion (department‑wise)
* [ ] Dockerfile & CI (GitHub Actions)

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

<p align="center">
  Built with ❤️ for students | Maintained by <a href="https://github.com/yourusername">@yourusername</a>
</p>
