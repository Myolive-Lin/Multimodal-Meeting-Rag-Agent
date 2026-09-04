# Enterprise Multimodal RAG Agent

A multimodal ingestion and retrieval-augmented generation (RAG) prototype designed for meeting knowledge bases. The system exposes a question-answering API through FastAPI, orchestrates agent tool calls with LangGraph, and uses PostgreSQL with pgvector for vector storage and similarity search.

> This repository is a functional engineering prototype, not a production-ready system. Images, video, and audio are converted to text before entering a unified text retrieval pipeline. The system does not use a vision-language model to interpret visual content directly.

## Key Features

- **Multimodal ingestion**: Processes video, image, subtitle, and plain-text files.
- **Video content extraction**: Samples video frames at a fixed interval and applies PaddleOCR, while FFmpeg extracts the audio track for transcription with faster-whisper.
- **Two-stage retrieval**: Retrieves candidates from pgvector with Sentence Transformer embeddings, then reranks them with a Cross-Encoder.
- **Agent orchestration**: Implements a model → tool → model execution loop with LangGraph.
- **HTTP API**: Exposes a `POST /chat` endpoint through FastAPI.
- **Observability foundation**: Includes Langfuse configuration and callback initialization, although the callback is not currently attached to API requests.

## Technology Stack

| Layer | Technology | Current responsibility |
| --- | --- | --- |
| API | FastAPI, Uvicorn | Exposes the question-answering endpoint |
| Agent | LangGraph, LangChain | Manages agent state, tool binding, and the execution loop |
| LLM | OpenAI-compatible API | Selects tools and generates the final response |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Produces normalized 384-dimensional vectors |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranks vector-search candidates by relevance |
| Vector store | PostgreSQL, pgvector | Stores document chunks and performs cosine-distance search |
| OCR | PaddleOCR | Extracts text from images and sampled video frames |
| ASR | faster-whisper, FFmpeg | Extracts and transcribes audio |
| Subtitles | pysubs2 | Parses SRT, VTT, ASS, and SSA subtitle files |
| Observability | Langfuse | Initialized, but not currently attached to the request path |

## Supported Data Types

| Type | Extensions | Processing method | Offline indexing |
| --- | --- | --- | --- |
| Video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv` | OCR on one frame every five seconds, plus audio transcription | Supported |
| Image | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif` | PaddleOCR | Supported |
| Subtitle | `.srt`, `.vtt`, `.ass`, `.ssa` | Parsed with pysubs2 and merged in five-second windows | Supported |
| Text | `.txt`, `.md`, `.log` | Loaded and recursively chunked | Supported |
| Standalone audio | Any format readable by FFmpeg/faster-whisper | On-demand transcription through `read_audio` only | Not yet supported |

Text files are split with `chunk_size=1000` and `chunk_overlap=150`. Extracted video, image, and subtitle content is written as documents without passing through this text splitter.

## Architecture

### Offline Indexing Pipeline

```mermaid
flowchart LR
    A["Local meeting assets"] --> B{"File type"}
    B -->|Video| C["Frame sampling + PaddleOCR"]
    B -->|Video audio| D["FFmpeg + faster-whisper"]
    B -->|Image| E["PaddleOCR"]
    B -->|Subtitle| F["pysubs2"]
    B -->|Text| G["RecursiveCharacterTextSplitter"]
    C --> H["Unified text Document"]
    D --> H
    E --> H
    F --> H
    G --> H
    H --> I["Sentence Transformer"]
    I --> J["PostgreSQL / pgvector"]
```

### Online Question-Answering Pipeline

```mermaid
flowchart LR
    A["POST /chat"] --> B["LangGraph Agent"]
    B --> C{"Call a tool?"}
    C -->|Search knowledge base| D["Embed query"]
    D --> E["Retrieve top 20 from pgvector"]
    E --> F["Cross-Encoder reranking; keep top 5"]
    F --> B
    C -->|Use a media tool| G["Read or ingest local media"]
    G --> B
    C -->|No tool needed| H["Generate final answer"]
    B --> H
    H --> I["JSON response"]
```

The agent currently registers five tools:

| Tool | Responsibility |
| --- | --- |
| `query_documents` | Runs vector retrieval and Cross-Encoder reranking |
| `read_audio` | Transcribes a specified audio or video file |
| `read_image` | Extracts text from a specified image with OCR |
| `read_video` | Applies OCR to video frames and transcribes the audio track |
| `ingest_media` | Indexes supported files from a specified directory |

## Repository Structure

```text
.
├── app/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Environment settings and defaults
│   ├── api/routes.py           # POST /chat route
│   ├── agent/agent.py          # LangGraph state graph and tool binding
│   ├── tool/
│   │   ├── parsers.py          # OCR, ASR, subtitle, and video parsing
│   │   └── rag_tools.py        # LangChain tool wrappers
│   ├── rag/
│   │   ├── ingest.py           # File discovery, loading, and text splitting
│   │   ├── embed.py            # Cached embedding and reranker models
│   │   ├── retriever.py        # Two-stage retrieval pipeline
│   │   └── store.py            # Database writes and vector search
│   └── db/schema.sql           # pgvector extension and documents table
├── job/build_index.py          # Offline indexing job
├── metting_notes/              # Default knowledge directory; existing spelling retained
├── pyproject.toml              # Project metadata and dependencies
└── uv.lock                     # Locked dependency versions
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- PostgreSQL with the pgvector extension installed
- FFmpeg for video and audio processing
- Access to an OpenAI-compatible Chat Completions service
- Network access for initial model downloads, or an existing local model cache

The repository includes a `uv.lock` file. The recommended setup is:

```bash
uv sync
```

### 2. Initialize the Database

Create the target database, then run:

```bash
psql postgresql://USER:PASSWORD@HOST:5432/DATABASE \
  -f app/db/schema.sql
```

This enables the `vector` extension and creates the `documents` table. The vector column is fixed at `VECTOR(384)`, so the embedding model must produce 384-dimensional vectors.

### 3. Configure the Environment

Create a `.env` file in the project root:

```dotenv
APP_NAME=enterprise-agent

# OpenAI-compatible LLM endpoint
LLM_MODEL=your-model-name
MODEL_URL=https://your-openai-compatible-endpoint/v1
TOKEN_API_KEY=your-api-key

# PostgreSQL / pgvector
POSTGRES_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE

# Retrieval models
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RAG_DIRECTORY=metting_notes

# Media processing
# Declared in settings, but the parser currently retains its 5-second default.
FRAME_INTERVAL=5.0
WHISPER_MODEL_SIZE=base
LANGUAGE=zh
DEVICE=cpu

# Langfuse initialization; tracing is not enabled in the current route.
LANGFUSE_PUBLIC_KEY=your-langfuse-public-key
LANGFUSE_SECRET_KEY=your-langfuse-secret-key
LANGFUSE_HOST=https://cloud.langfuse.com
```

Never commit real credentials or database passwords. In production, inject secrets through a secrets manager or the deployment platform.

### 4. Build the Knowledge Index

Run from the project root:

```bash
uv run python -m job.build_index
```

The job scans `RAG_DIRECTORY`, extracts text, produces normalized embeddings, and inserts each document into the `documents` table. A successful run prints:

```text
Indexed N chunks.
```

> The current indexing job does not provide deduplication, replacement, or incremental updates. Re-running it inserts duplicate records.

### 5. Start the API

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Add `--reload` for local development. The following endpoints are then available:

- API: `http://127.0.0.1:8000`
- OpenAPI UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 6. Send a Question

```bash
curl --request POST 'http://127.0.0.1:8000/chat' \
  --header 'Content-Type: application/json' \
  --data '{"messages":"What are the next actions agreed upon in the project meeting?"}'
```

Example response:

```json
{
  "answer": "..."
}
```

## API Reference

### `POST /chat`

Request body:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `messages` | `string` | Yes | A single-turn user question. Despite its plural name, it does not accept a message array. |

Successful response:

| Field | Type | Description |
| --- | --- | --- |
| `answer` | `string` | Text content of the final agent message |

The endpoint currently has no authentication, streaming response, conversation identifier, structured error model, or rate limiting.

## Configuration Reference

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `LLM_MODEL` | Yes | — | Model name exposed by the OpenAI-compatible service |
| `MODEL_URL` | Yes | — | Base URL of the model service |
| `TOKEN_API_KEY` | Yes | — | Model service credential |
| `POSTGRES_URL` | Yes | — | SQLAlchemy PostgreSQL connection URL |
| `APP_NAME` | No | `enterprise-agent` | Reserved name; the FastAPI title is currently hard-coded |
| `EMBED_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | 384-dimensional embedding model |
| `RERANKER_MODEL` | No | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-Encoder reranking model |
| `RAG_DIRECTORY` | No | `metting_notes` | Directory scanned by the offline indexing job |
| `FRAME_INTERVAL` | No | `5.0` | Reserved setting; the parser call still uses a five-second default |
| `WHISPER_MODEL_SIZE` | No | `base` | faster-whisper model size |
| `LANGUAGE` | No | `zh` | ASR language code |
| `DEVICE` | No | `cpu` | faster-whisper inference device |
| `LANGFUSE_PUBLIC_KEY` | See note | `None` | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | See note | `None` | Langfuse secret key |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse server URL |

`LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are typed as optional, but the application writes them to environment variables and initializes `CallbackHandler` during import. Missing values may therefore prevent startup and should be treated as runtime requirements until initialization is conditional. The callback argument in the [API route](app/api/routes.py) is also commented out, so ordinary `/chat` requests do not produce complete Langfuse traces.

## Retrieval Design

The retrieval pipeline follows a standard retrieve–rerank pattern:

1. `all-MiniLM-L6-v2` encodes the question as a normalized 384-dimensional vector.
2. pgvector uses the `<=>` cosine-distance operator to retrieve up to 20 candidate chunks.
3. `ms-marco-MiniLM-L-6-v2` scores each `(question, candidate text)` pair.
4. The five highest-scoring candidates are concatenated and returned to the agent as tool context.

The current query path does not implement a relevance threshold, metadata filtering, hybrid retrieval, or context-length control.

## Known Limitations and Security Boundaries

- **Prototype API**: Supports synchronous, single-turn questions only. It has no persistent conversations, streaming, authentication, rate limiting, or tenant isolation.
- **Local file access**: Media tools accept filesystem paths without a path allowlist or access-control layer. Do not expose the service directly to untrusted networks.
- **Directory-level ingestion**: When `ingest_media` receives a file path, it scans the file's entire parent directory rather than only that file.
- **Standalone audio indexing**: `read_audio` can transcribe audio on demand, but offline `load_and_split` does not recognize audio extensions.
- **Partially wired settings**: `FRAME_INTERVAL` and `APP_NAME` are declared but do not override the corresponding hard-coded behavior.
- **Index consistency**: There is no content hash, uniqueness constraint, incremental synchronization, or deletion mechanism. Repeated indexing creates duplicate records.
- **Embedding dimension coupling**: Changing the embedding model requires a schema migration and a full reindex.
- **Retrieval performance**: The IVFFlat index in `schema.sql` is commented out. As the corpus grows, vector queries may fall back to a sequential scan.
- **Error handling**: Some media parsing failures become empty results, and the API does not define a consistent error response model.
- **Langfuse status**: The callback is initialized but not attached to `/chat`; observability is incomplete.
- **Testing and delivery**: The repository does not yet include automated tests, a container image, CI/CD, health checks, or a production deployment guide.

## Production Readiness Roadmap

Recommended priorities:

1. Restrict filesystem access and add authentication, authorization, rate limiting, and audit logging.
2. Introduce stable document identifiers, content hashes, and uniqueness constraints for idempotent incremental indexing and deletion synchronization.
3. Move OCR, ASR, and indexing workloads to an asynchronous queue with job status, retries, and failure recovery.
4. Enable the Langfuse callback and add structured logs, metrics, trace IDs, and sensitive-data redaction.
5. Add relevance thresholds, metadata filtering, hybrid retrieval, and an offline evaluation dataset.
6. Add unit, integration, and API contract tests, followed by Docker packaging, CI/CD, and health checks.

## Suggested Reading Order

1. `app/api/routes.py` — HTTP request entry point.
2. `app/agent/agent.py` — LangGraph state graph and tool execution loop.
3. `app/tool/rag_tools.py` — tools available to the agent.
4. `app/rag/retriever.py` — vector retrieval and reranking.
5. `app/rag/store.py` — database writes and vector queries.
6. `app/rag/ingest.py` and `app/tool/parsers.py` — multimodal-to-text processing.
7. `job/build_index.py` — offline indexing entry point.

## Project Status

The project implements the full prototype path from multimodal-to-text extraction through vector indexing, two-stage retrieval, agent tool execution, and HTTP response generation. It is suitable for technical validation, demonstrations, and small-scale internal experiments. Complete the security, reliability, observability, and data-governance work above before considering a production deployment.
