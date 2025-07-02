# WikiBot

WikiBot is a simple command line application that demonstrates an autonomous research agent using Wikipedia and an open-source language model via Hugging Face.

## Features
- Decides whether to search Wikipedia or answer directly.
- Loops up to five reasoning steps using a ReAct style prompt.
- Fetches Wikipedia summaries via the public API.
- Uses the `mistralai/Mistral-7B-Instruct-v0.1` model through the Hugging Face Inference API.
- Logs each step and can optionally save logs to a text file.
- Provides a CLI for single questions or an interactive session.

## Requirements
- Python 3.8+
- `requests` (see `requirements.txt`)
- A Hugging Face API token with access to the chosen model.

## Usage
Install dependencies:
```bash
pip install -r requirements.txt
```

Run WikiBot with a question:
```bash
export HF_TOKEN=your_hf_token
python wikibot.py "What are the latest discoveries about Mars?"
```

For an interactive session:
```bash
python wikibot.py
```

Use `--log session.txt` to save the reasoning trace to a file.
