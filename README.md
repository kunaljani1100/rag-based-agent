# RAG-Based Agent

A minimal Retrieval-Augmented Generation (RAG) agent that combines local semantic search with Claude to answer questions grounded in a small document set.

## How It Works

1. A fixed list of short text documents is embedded using a `sentence-transformers` model (`all-MiniLM-L6-v2`).
2. When a query comes in, it's embedded the same way, and cosine similarity is used to retrieve the top-k most relevant documents.
3. The retrieved documents are injected into a prompt as context, which is sent to Claude (`claude-sonnet-4-6`) via the Anthropic API.
4. Claude answers the question using only the provided context, and explicitly says so if the context doesn't contain the answer.

## Requirements

- Python 3.9+
- [Anthropic API key](https://console.anthropic.com/)

### Dependencies

```bash
pip install anthropic sentence-transformers numpy
```

## Setup

1. Clone the repo:

```bash
   git clone https://github.com/kunaljani1100/rag-based-agent.git
   cd rag-based-agent
```

2. Install dependencies (see above).

3. Set your Anthropic API key. Rather than hardcoding it in the script, export it as an environment variable:

```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
```

   And update the client initialization to:

```python
   client = anthropic.Anthropic()
```

   (The SDK will automatically pick up `ANTHROPIC_API_KEY` from the environment.)

## Usage

Run the script and enter a question when prompted:

```bash
python rag-implementation.py
```

Example: What is the purpose of GKE?
