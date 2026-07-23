"""
Task 2 — OpenRouter LLM Integration
=====================================

This module adds a natural-language layer on top of Task 1's keyword
search: it takes the FAQ that best matches a user's question and asks an
LLM (via OpenRouter) to rephrase the official answer conversationally,
while staying strictly grounded in that FAQ's content.

This module REUSES Task 1's `FAQS` list and `search_by_keyword()` function
rather than redefining them — see the import below.

Setup
-----
1. Install the one dependency this module needs:
       pip install requests
2. Set your OpenRouter API key as an environment variable (never hard-code
   it in source):
       export OPENROUTER_API_KEY="sk-or-..."
3. Run this file directly to see the demonstration:
       python task2_llm_integration.py

Model
-----
Default model: "openai/gpt-4o-mini" (fast, cheap, good instruction
following — well suited to a short grounded-rephrasing task like this).
Any other OpenRouter-hosted chat model can be passed to LLMClient instead.
"""

import os
import requests
from typing import Dict, Optional

# Reuse Task 1's knowledge base and search function instead of redefining them.
from task1_faq_search import FAQS, search_by_keyword

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class LLMClientError(Exception):
    """Raised when the OpenRouter API call fails for any reason."""


class LLMClient:
    """
    A thin wrapper around the OpenRouter chat-completions API.

    Handles building the request, calling the API, and turning both HTTP
    errors and network errors into a single, clearly-worded exception so
    callers don't need to know about `requests` internals.
    """

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        """
        Args:
            api_key: OpenRouter API key. Should come from an environment
                variable (see `OPENROUTER_API_KEY`), never hard-coded.
            model: OpenRouter model identifier, e.g. "openai/gpt-4o-mini".
        """
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        max_tokens: int = 512,
    ) -> str:
        """
        Send a chat completion request to OpenRouter and return the
        assistant's reply text.

        Args:
            prompt: the user-role message content.
            system_message: optional system-role instruction to steer
                the model's behaviour (tone, grounding rules, etc.).
            max_tokens: maximum number of tokens in the completion.

        Returns:
            The assistant's response text, stripped of leading/trailing
            whitespace.

        Raises:
            LLMClientError: if the request fails (network problem, bad
                API key, rate limit, malformed response, etc.). The
                message is descriptive enough to show the user/log.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=30
            )
            # Raises requests.HTTPError for 4xx/5xx status codes.
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            # Try to surface the API's own error message if it provided one.
            detail = ""
            try:
                detail = response.json().get("error", {}).get("message", "")
            except Exception:
                detail = response.text[:200]
            raise LLMClientError(
                f"OpenRouter API request failed with status "
                f"{response.status_code}: {detail or http_err}"
            ) from http_err
        except requests.exceptions.ConnectionError as conn_err:
            raise LLMClientError(
                f"Could not reach OpenRouter API (network/connection error): {conn_err}"
            ) from conn_err
        except requests.exceptions.Timeout as timeout_err:
            raise LLMClientError(
                f"OpenRouter API request timed out: {timeout_err}"
            ) from timeout_err
        except requests.exceptions.RequestException as req_err:
            raise LLMClientError(
                f"OpenRouter API request failed: {req_err}"
            ) from req_err

        try:
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as parse_err:
            raise LLMClientError(
                f"Unexpected response format from OpenRouter API: {parse_err}"
            ) from parse_err

    def generate_faq_response(self, user_question: str, faq_entry: Dict) -> str:
        """
        Rephrase an FAQ's official answer into a natural, conversational
        reply to the user's specific question — while staying grounded
        strictly in that FAQ's content.

        Prompt design (see module docstring / README section below for the
        full rationale):
          - The SYSTEM prompt fixes the model's role, tone, length limit,
            and — most importantly — a hard grounding rule: only use facts
            present in the supplied FAQ, and say so plainly if the FAQ
            doesn't fully cover what was asked, rather than inventing
            details.
          - The USER prompt supplies the FAQ question, the official
            answer, and the user's actual question as clearly labelled,
            separate fields, so the model can't confuse "what the user
            asked" with "what the FAQ says".

        Args:
            user_question: the raw question typed by the end user.
            faq_entry: an FAQ dict (as defined in Task 1) containing at
                least "question" and "answer" fields.

        Returns:
            The LLM-generated, conversational response text.

        Raises:
            LLMClientError: propagated from generate() if the API call fails.
        """
        system_message = (
            "You are SupportAI, a friendly and professional customer support "
            "agent. You must answer the user's question using ONLY the "
            "information contained in the FAQ_QUESTION and FAQ_ANSWER provided "
            "below — do not invent, assume, or add any facts, numbers, links, "
            "or policies that are not explicitly stated there. "
            "If the FAQ content does not fully answer what the user is asking, "
            "clearly say that you don't have that specific detail and "
            "recommend they contact human support, rather than guessing. "
            "Keep your response friendly, natural, and professional in tone, "
            "written in plain conversational language (not a robotic copy of "
            "the FAQ text). Keep it under 150 words."
        )

        prompt = (
            f"FAQ_QUESTION: {faq_entry['question']}\n"
            f"FAQ_ANSWER: {faq_entry['answer']}\n\n"
            f"USER_QUESTION: {user_question}\n\n"
            "Using only the FAQ_ANSWER above, write a natural, conversational "
            "reply to the USER_QUESTION."
        )

        return self.generate(prompt, system_message=system_message, max_tokens=200)


# ---------------------------------------------------------------------------
# Demonstration
# ---------------------------------------------------------------------------

def answer_user_question(client: LLMClient, user_question: str) -> None:
    """
    Run the full pipeline for one user question and print the result:
        1. Find the best-matching FAQ using Task 1's search_by_keyword().
        2. Pass it to generate_faq_response() to get a conversational reply.
        3. Print the question, matched FAQ, and generated response.

    Args:
        client: a configured LLMClient instance.
        user_question: the question typed by the end user.
    """
    print(f"Question: {user_question}")

    matches = search_by_keyword(FAQS, user_question)
    if not matches:
        print("Matched FAQ: none found")
        print("SupportAI Response:\n  Sorry, I couldn't find anything relevant "
              "in our FAQs. Please contact human support for help.\n")
        return

    best_match = matches[0]  # highest-ranked match
    print(f"Matched FAQ: {best_match['question']}")

    try:
        reply = client.generate_faq_response(user_question, best_match)
        print(f"SupportAI Response:\n{reply}\n")
    except LLMClientError as err:
        # Graceful, descriptive failure instead of a raw traceback.
        print(f"SupportAI Response: [LLM call failed] {err}\n")


if __name__ == "__main__":
    # API key must come from an environment variable, never hard-coded.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "WARNING: OPENROUTER_API_KEY is not set. Set it with:\n"
            "  export OPENROUTER_API_KEY=\"sk-or-...\"\n"
            "Proceeding anyway to demonstrate error handling.\n"
        )
        api_key = "missing-key"  # will trigger a clean auth error from the API

    llm_client = LLMClient(api_key=api_key, model=DEFAULT_MODEL)

    demo_questions = [
        "I can't remember my login password",
        "Can I get my money back after buying a subscription?",
    ]

    for question in demo_questions:
        answer_user_question(llm_client, question)