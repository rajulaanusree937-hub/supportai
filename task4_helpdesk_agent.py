"""
Task 4 — Complete Helpdesk Agent
===================================

This is the final integration layer of SupportAI. It combines:
    - Task 1: FAQ knowledge base + keyword search
    - Task 2: LLMClient for natural-language, grounded FAQ responses
    - Task 3: hybrid_search() (keyword + TF-IDF) for robust FAQ matching

into a `SupportAgent` that can hold a multi-turn conversation, track
confidence per turn, fall back gracefully on out-of-scope questions, and
escalate to a mock human-support ticket on request.

Nothing from Tasks 1-3 is reimplemented here — this module only imports
and orchestrates it.

Setup
-----
    pip install requests scikit-learn
    export OPENROUTER_API_KEY="sk-or-v1-..."   (Linux/Mac)
    $env:OPENROUTER_API_KEY="sk-or-v1-..."     (Windows PowerShell)

Run
---
    python task4_helpdesk_agent.py
"""

import os
import random
from dataclasses import dataclass
from typing import List, Optional

# Reuse everything from Tasks 1-3 rather than reimplementing it.
from task1_faq_search import FAQS
from task2_llm_integration import LLMClient, LLMClientError, DEFAULT_MODEL
from task3_faq_matching import hybrid_search


# ---------------------------------------------------------------------------
# Conversation turn record
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """One message in the conversation, either from the user or the agent."""
    role: str        # "user" or "assistant"
    content: str
    faq_id: str = None
    confidence: float = None


# ---------------------------------------------------------------------------
# SupportAgent — orchestrates Tasks 1-3
# ---------------------------------------------------------------------------

class SupportAgent:
    """
    Conversational helpdesk agent.

    Delegates FAQ matching to `hybrid_search()` (Task 3) and response
    generation to `LLMClient.generate_faq_response()` (Task 2). Tracks
    conversation history, confidence per turn, and a running streak of
    low-confidence answers so the chat interface can proactively suggest
    escalation.
    """

    LOW_CONFIDENCE_STREAK_LIMIT = 3

    def __init__(self, faqs: List[dict], llm_client: LLMClient, confidence_threshold: float = 0.15):
        """
        Args:
            faqs: FAQ knowledge base (Task 1's `FAQS`).
            llm_client: a configured LLMClient instance (Task 2).
            confidence_threshold: minimum hybrid_search score required to
                treat a match as confident enough to answer from.
        """
        self.faqs = faqs
        self.llm_client = llm_client
        self.confidence_threshold = confidence_threshold

        self.history: List[ConversationTurn] = []
        self.escalated = False
        self.low_confidence_streak = 0

    def handle_message(self, user_message: str) -> str:
        """
        Process one user message end-to-end: record it, find the best FAQ
        match via hybrid search, generate a grounded response (or a
        fallback if nothing confident was found), record the assistant's
        turn, and return the response text.

        Args:
            user_message: the raw text typed by the user.

        Returns:
            The agent's response text.
        """
        self.history.append(ConversationTurn(role="user", content=user_message))

        # Delegate matching entirely to Task 3's hybrid search.
        results = hybrid_search(self.faqs, user_message, top_k=1)
        best_faq, best_score = results[0] if results else (None, 0.0)

        if best_faq is not None and best_score >= self.confidence_threshold:
            # Confident match: ask the LLM (Task 2) to phrase a grounded reply.
            try:
                response_text = self.llm_client.generate_faq_response(user_message, best_faq)
            except LLMClientError as err:
                # Graceful degradation: fall back to the FAQ's raw answer
                # rather than crashing the conversation if the LLM call fails.
                response_text = (
                    f"{best_faq['answer']} "
                    f"(Note: natural-language rephrasing unavailable right now: {err})"
                )

            self.low_confidence_streak = 0  # reset streak on a confident answer
            self.history.append(
                ConversationTurn(
                    role="assistant",
                    content=response_text,
                    faq_id=best_faq["id"],
                    confidence=best_score,
                )
            )
        else:
            # No confident match: fallback response, offer escalation.
            response_text = (
                f"I don't have information about that in my knowledge base. "
                f"Would you like me to connect you with a human support agent?"
            )
            self.low_confidence_streak += 1
            self.history.append(
                ConversationTurn(
                    role="assistant",
                    content=response_text,
                    faq_id=None,
                    confidence=0.0,
                )
            )

        return response_text

    def should_suggest_escalation(self) -> bool:
        """
        Returns:
            True if the last `LOW_CONFIDENCE_STREAK_LIMIT` consecutive
            answers were all low-confidence, signalling the chat interface
            should proactively suggest escalation.
        """
        return self.low_confidence_streak >= self.LOW_CONFIDENCE_STREAK_LIMIT

    def escalate(self, reason: str = "User requested human support") -> str:
        """
        Mark the session as escalated and produce a mock support ticket.

        Args:
            reason: why the escalation happened (stored for the summary).

        Returns:
            A confirmation message including a mock ticket ID and an
            estimated response time.
        """
        self.escalated = True
        ticket_id = f"TICKET-{random.randint(10000, 99999)}"

        self.history.append(
            ConversationTurn(role="assistant", content=f"Escalated: {reason} ({ticket_id})")
        )

        return (
            "Your request has been escalated to our support team.\n"
            f"Ticket ID: {ticket_id}\n"
            "Estimated response time: within 4 business hours."
        )

    def get_conversation_summary(self) -> str:
        """
        Build a formatted summary of the entire conversation so far,
        including which FAQ (if any) and confidence score backed each
        assistant response.

        Returns:
            A multi-line formatted string. If there's no history yet,
            returns a short "no conversation" message instead.
        """
        if not self.history:
            return "No conversation yet."

        lines = ["Conversation Summary", "=" * 21]
        for turn in self.history:
            if turn.role == "user":
                lines.append(f"You: {turn.content}")
            else:
                detail = ""
                if turn.faq_id is not None:
                    detail = f" [faq: {turn.faq_id}, confidence: {turn.confidence:.2f}]"
                elif turn.confidence is not None:
                    detail = f" [confidence: {turn.confidence:.2f}]"
                lines.append(f"SupportAI:{detail} {turn.content}")

        lines.append("=" * 21)
        lines.append(f"Escalated: {self.escalated}")
        return "\n".join(lines)

    def reset(self) -> None:
        """Clear conversation history and reset escalation/streak state."""
        self.history = []
        self.escalated = False
        self.low_confidence_streak = 0


# ---------------------------------------------------------------------------
# Interactive chat interface
# ---------------------------------------------------------------------------

WELCOME_BANNER = (
    "╔══════════════════════════════════════════╗\n"
    "║       SupportAI — Helpdesk Agent          ║\n"
    "╚══════════════════════════════════════════╝"
)


def print_response(response: str, faq_id: Optional[str], confidence: Optional[float]) -> None:
    """
    Print an assistant response with its confidence/FAQ-id header, matching
    the format: 'SupportAI [confidence: 0.42, faq: faq-001]: ...'
    """
    if faq_id is not None:
        header = f"SupportAI [confidence: {confidence:.2f}, faq: {faq_id}]:"
    elif confidence is not None:
        header = f"SupportAI [confidence: {confidence:.2f}]:"
    else:
        header = "SupportAI:"
    print(f"\n{header}\n{response}\n")


def run_chat_interface(agent: SupportAgent) -> None:
    """
    Run the interactive command-line chat loop.

    Supported inputs:
        <any text>  -> sent to the agent as a question
        "history"   -> print conversation summary
        "escalate"  -> trigger escalation
        "reset"     -> clear conversation and start fresh
        "quit"      -> exit
    """
    print(WELCOME_BANNER)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        command = user_input.lower()

        if command == "quit":
            print("Thank you for using SupportAI. Goodbye!")
            break

        elif command == "history":
            print(f"\n{agent.get_conversation_summary()}")

        elif command == "escalate":
            confirmation = agent.escalate()
            print(f"\nSupportAI:\n{confirmation}")

        elif command == "reset":
            agent.reset()
            print("\nConversation has been reset. Starting fresh!")

        else:
            response = agent.handle_message(user_input)
            last_turn = agent.history[-1]
            print_response(response, last_turn.faq_id, last_turn.confidence)

            if agent.should_suggest_escalation():
                print(
                    "SupportAI: It looks like I'm having trouble finding what you need. "
                    "Would you like to type 'escalate' to talk to a human agent?\n"
                )


# ---------------------------------------------------------------------------
# End-to-end demonstration (non-interactive, scripted)
# ---------------------------------------------------------------------------

def run_scripted_demo() -> None:
    """
    Run the 4 required demo scenarios non-interactively, printing each
    step, so the whole pipeline can be verified without manual typing:
        1. Clear FAQ question -> high-confidence answer.
        2. Paraphrased question -> matched via hybrid search.
        3. Out-of-scope question -> fallback + escalation offer.
        4. Escalation request -> mock ticket confirmation.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "missing-key")
    llm_client = LLMClient(api_key=api_key, model=DEFAULT_MODEL)
    agent = SupportAgent(FAQS, llm_client, confidence_threshold=0.15)

    print(WELCOME_BANNER)

    scripted_turns = [
        "How do I change my password?",              # 1. clear FAQ question
        "I forgot my login credentials",              # 2. paraphrased question
        "What are your office hours in Tokyo?",        # 3. out-of-scope question
        "escalate",                                    # 4. escalation request
    ]

    for user_input in scripted_turns:
        print(f"\nYou: {user_input}")

        if user_input.lower() == "escalate":
            print(f"\nSupportAI:\n{agent.escalate()}")
            continue

        response = agent.handle_message(user_input)
        last_turn = agent.history[-1]
        print_response(response, last_turn.faq_id, last_turn.confidence)

        if agent.should_suggest_escalation():
            print(
                "SupportAI: It looks like I'm having trouble finding what you need. "
                "Would you like to type 'escalate' to talk to a human agent?\n"
            )

    print("You: quit")
    print("Thank you for using SupportAI. Goodbye!\n")

    print(agent.get_conversation_summary())


if __name__ == "__main__":
    # Run the scripted 4-scenario demo first so grading/output is
    # reproducible without manual input...
    run_scripted_demo()

    # ...then drop into the real interactive chat if a real terminal is
    # attached (skips automatically if run non-interactively, e.g. in CI).
    import sys
    if sys.stdin.isatty():
        print("\n\nStarting interactive chat — type 'quit' to exit.\n")
        api_key = os.environ.get("OPENROUTER_API_KEY", "missing-key")
        llm_client = LLMClient(api_key=api_key, model=DEFAULT_MODEL)
        interactive_agent = SupportAgent(FAQS, llm_client, confidence_threshold=0.15)
        run_chat_interface(interactive_agent)