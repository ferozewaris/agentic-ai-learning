"""WikiBot: an autonomous research agent that uses Wikipedia and an open-source LLM.

This script implements a simple loop where an LLM decides whether to search
Wikipedia or return an answer directly. The agent communicates in a
ReAct-like format using `ACTION:` and `ANSWER:` tokens.

Dependencies: only Python standard library and `requests`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import requests

# Constants for the Hugging Face model
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

@dataclass
class StepLog:
    """Stores information about a single reasoning step."""

    step: int
    thought: str
    action: Optional[str] = None
    observation: Optional[str] = None


@dataclass
class WikiBot:
    """A minimal autonomous research agent for Wikipedia."""

    token: str
    max_steps: int = 5
    log_to_file: Optional[str] = None
    logs: List[StepLog] = field(default_factory=list)

    def log(self, entry: StepLog) -> None:
        self.logs.append(entry)
        if self.log_to_file:
            with open(self.log_to_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.__dict__) + "\n")

    def build_prompt(self, question: str, observation: Optional[str]) -> str:
        """Constructs the prompt for the LLM."""
        prompt_parts = [
            "You are WikiBot, an autonomous research agent that can answer questions using Wikipedia.",
            f"The question is: '{question}'.",
            "You may respond with either:\n",
            "ACTION: <search query> -- to search Wikipedia for information",
            "ANSWER: <final answer> -- when you know the answer.",
        ]
        if observation:
            prompt_parts.append(f"Tool result: {observation}")
        prompt_parts.append("Think step by step and decide your next action or final answer.")
        return "\n".join(prompt_parts)

    def call_llm(self, prompt: str) -> str:
        """Calls the Hugging Face Inference API and returns the raw text."""
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 200, "temperature": 0.2}}
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "").strip()
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        return str(data)

    def parse_response(self, text: str) -> (str, str):
        """Parses the LLM response into (type, content)."""
        cleaned = text.strip()
        if cleaned.upper().startswith("ACTION:"):
            return "action", cleaned[len("ACTION:"):].strip()
        if cleaned.upper().startswith("ANSWER:"):
            return "answer", cleaned[len("ANSWER:"):].strip()
        # Fallback: treat anything else as a final answer
        return "answer", cleaned

    def search_wikipedia(self, query: str) -> str:
        """Fetches the top Wikipedia summary for a query."""
        search_url = "https://en.wikipedia.org/w/api.php"
        search_params = {
            "action": "query",
            "list": "search",
            "format": "json",
            "srsearch": query,
        }
        search_resp = requests.get(search_url, params=search_params, timeout=30)
        search_resp.raise_for_status()
        data = search_resp.json()
        results = data.get("query", {}).get("search", [])
        if not results:
            return "No results found."
        top_title = results[0]["title"]
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{top_title}"
        summary_resp = requests.get(summary_url, timeout=30)
        summary_resp.raise_for_status()
        summary_data = summary_resp.json()
        return summary_data.get("extract", "No summary available.")

    def run(self, question: str) -> str:
        """Runs the reasoning loop for the provided question."""
        observation = None
        for step in range(1, self.max_steps + 1):
            prompt = self.build_prompt(question, observation)
            response = self.call_llm(prompt)
            decision, content = self.parse_response(response)
            log_entry = StepLog(step=step, thought=response)
            if decision == "action":
                log_entry.action = content
                observation = self.search_wikipedia(content)
                log_entry.observation = observation
                self.log(log_entry)
            else:
                log_entry.action = None
                self.log(log_entry)
                return content
        return "Reached step limit without a final answer."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run WikiBot, an autonomous research agent.")
    parser.add_argument("question", nargs="?", help="Question for WikiBot to answer")
    parser.add_argument("--token", required=False, help="Hugging Face API token (or set HF_TOKEN env var)")
    parser.add_argument("--log", help="Optional path to log file")
    args = parser.parse_args()

    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("Hugging Face API token required via --token or HF_TOKEN env var")

    if args.question:
        bot = WikiBot(token=token, log_to_file=args.log)
        answer = bot.run(args.question)
        print("Final Answer:\n", answer)
        print("\n--- Trace ---")
        for entry in bot.logs:
            print(f"Step {entry.step}: {entry.thought}")
            if entry.action:
                print(f"  Action: {entry.action}")
            if entry.observation:
                print(f"  Observation: {entry.observation[:200]}...")
    else:
        # interactive mode
        bot = WikiBot(token=token, log_to_file=args.log)
        print("Enter a question (empty line to exit):")
        while True:
            q = input("? ").strip()
            if not q:
                break
            bot.logs.clear()
            ans = bot.run(q)
            print("Answer:", ans)
            print("--- Trace ---")
            for entry in bot.logs:
                print(f"Step {entry.step}: {entry.thought}")
                if entry.action:
                    print(f"  Action: {entry.action}")
                if entry.observation:
                    print(f"  Observation: {entry.observation[:200]}...")


if __name__ == "__main__":
    main()
