# Funes Semantic Memory Spec — Context-Window Injection (v0.1.0)

## Overview

Funes extends llama.cpp with persistent semantic memory. User facts are embedded,
stored in LogosDB, and retrieved at query time via HNSW nearest-neighbor search.
Retrieved memory is injected into the prompt context window as a Quote or Hint
suffix, enabling factual recall without model fine-tuning.

This approach is **model-agnostic** — it works with any architecture supported by
llama.cpp because it operates at the prompt level, before tokenization.

## Architecture

```
User input
  │
  ├─ /teach or <teach> tags ──► generate_embedding() ──► logosdb_put()
  │
  └─ normal message
       │
       ├─ generate_embedding(query)
       │      │
       │      └─► logosdb_search(top_k) ──► memory chunks with text + timestamp
       │             │
       │             └─► LLM compressor decides:
       │                   Quote: "exact text"   ← short precise fact
       │                   Hint: summary          ← needs interpretation
       │                   NONE                   ← irrelevant
       │
       ├─ append suffix to user message
       │
       └─► main LLM completion
```

## Storage: LogosDB

Memory is stored in a LogosDB database (HNSW index + mmap vectors + JSONL metadata).

- **Write**: `logosdb_put(db, embedding, dim, text, timestamp, &err)`
- **Search**: `logosdb_search(db, query, dim, top_k, &err)` — O(log n) via HNSW
- **Persistence**: three files in the DB directory: `vectors.bin`, `meta.jsonl`, `hnsw.idx`
- **Scale**: tested to 100K vectors; designed for millions

Replaces the previous flat-file format (space-separated floats + `.meta.jsonl` sidecar).

## Dual injection: Quote vs Hint

The memory compressor (a short LLM completion) decides which injection mode to use:

| Mode | When | Example |
|------|------|---------|
| **Quote** | Memory contains a short, precise fact that directly answers the query | `Quote: "My commute is 42 minutes"` |
| **Hint** | Memory is relevant but needs interpretation or combination | `Hint: The user's daily commute takes about 42 minutes` |
| **NONE** | No memory is relevant | (nothing injected) |

The injection is appended as a suffix to the user message before sending to the model.

**Fallback**: When `--semantic-memory-hint-llm-compress` is off, a heuristic token
extraction (`extract_hint_tokens`) produces a keyword-based `Hint:` suffix instead.

## CLI flags

| Flag | Description |
|------|-------------|
| `--semantic-memory-db <path>` | Path to LogosDB database directory |
| `--semantic-memory-dim <D>` | Embedding dimension (required with `--semantic-memory-db`) |
| `--semantic-memory-hint-tokens <N>` | Heuristic keyword hint token count (default: 0, disabled) |
| `--semantic-memory-hint-llm-compress` | Enable LLM compressor for Quote/Hint |
| `--no-semantic-memory-hint-llm-compress` | Disable LLM compressor |
| `--semantic-memory-hint-llm-top-k <K>` | Top-k memory chunks for compressor (default: 3) |
| `--semantic-memory-hint-llm-n-predict <N>` | Max tokens for compressor output (default: 48) |
| `--semantic-memory-teach-tags` | Enable `<teach>...</teach>` inline tags (default: on) |
| `--no-semantic-memory-teach-tags` | Disable teach tags |
| `--semantic-memory-teach-open-tag <TAG>` | Custom teach open tag (default: `<teach>`) |
| `--semantic-memory-teach-close-tag <TAG>` | Custom teach close tag (default: `</teach>`) |
| `--semantic-memory-teach-prefix <PREFIX>` | Treat messages starting with prefix as teach-only |
| `--semantic-memory-teach-variants` | Generate paraphrase variants when teaching |
| `--no-semantic-memory-teach-variants` | Disable teach variants |
| `--semantic-memory-learn-response-tags` | Learn from assistant responses via tags (default: on) |
| `--no-semantic-memory-learn-response-tags` | Disable learn-from-response |
| `--semantic-memory-learn-response-open-tag <TAG>` | Custom learn-from-response open tag (default: `<learn-from-response>`) |
| `--semantic-memory-learn-response-close-tag <TAG>` | Custom learn-from-response close tag (default: `</learn-from-response>`) |

## Supported architectures (top 10)

Context-window injection is model-agnostic. The following architecture families
are explicitly supported and tested:

| Family | Provider | Model files |
|--------|----------|-------------|
| LLaMA | Meta | `llama.cpp`, `llama-iswa.cpp` |
| Qwen | Alibaba | `qwen.cpp`, `qwen2.cpp`, `qwen3.cpp` |
| Gemma | Google | `gemma.cpp`, `gemma3.cpp`, `gemma4-iswa.cpp` |
| Phi | Microsoft | `phi2.cpp`, `phi3.cpp` |
| Mistral | Mistral AI | `mistral3.cpp` |
| DeepSeek | DeepSeek | `deepseek.cpp`, `deepseek2.cpp` |
| Command-R | Cohere | `command-r.cpp` |
| GPT-2 / GPT-NeoX | OpenAI / EleutherAI | `gpt2.cpp`, `gptneox.cpp` |
| Falcon | TII | `falcon.cpp`, `falcon-h1.cpp` |
| LFM | Liquid AI | `lfm2.cpp` |
| Mamba | state-space | `mamba.cpp`, `mamba-base.cpp` |

Plus 100+ additional architectures supported by llama.cpp.

## Temporal context

The compressor receives pre-computed date arithmetic for any ISO 8601 dates found
in memory chunks and the query. This avoids LLM date-math errors.

## Dependencies

- **LogosDB v0.3.0** — fetched via CMake FetchContent from
  [github.com/jose-compu/logosdb](https://github.com/jose-compu/logosdb)
  - Batch put API for bulk ingestion (~4× faster)
  - Delete and update operations
  - Soft-delete with live/dead row tracking
- **llama.cpp** — forked in-tree with ggml backend
