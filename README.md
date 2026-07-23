# SupportAI — Conversational Helpdesk Agent

SupportAI is a small end-to-end helpdesk assistant built in four stages:
a keyword-searchable FAQ knowledge base, an LLM layer that rephrases
answers conversationally, a smarter TF-IDF/hybrid matcher for paraphrased
questions, and finally a full conversational agent that ties it all
together with escalation to human support.

## Project Structure

```
supportai/
├── task1_faq_search.py       # FAQ knowledge base + keyword search
├── task2_llm_integration.py  # OpenRouter LLM client + grounded responses
├── task3_faq_matching.py     # TF-IDF matching + hybrid search
├── task4_helpdesk_agent.py   # Full conversational agent + chat interface
├── .gitignore
└── README.md
```

Each file builds on the previous one — nothing is reimplemented, only
imported and extended.

## Setup

1. **Clone the repo** and move into it:
   ```bash
   git clone https://github.com/rajulaanusree937-hub/supportai.git
   cd supportai
   ```

2. **Install dependencies**:
   ```bash
   python -m pip install requests scikit-learn
   ```

3. **Set your OpenRouter API key** (required for Tasks 2 and 4 to get
   real LLM-generated responses; never hard-coded in source):

   Windows PowerShell:
   ```bash
   $env:OPENROUTER_API_KEY="sk-or-v1-your-real-key"
   ```
   Linux / macOS:
   ```bash
   export OPENROUTER_API_KEY="sk-or-v1-your-real-key"
   ```

   Get a key at https://openrouter.ai/keys (requires a small amount of
   account credit to call paid models — see https://openrouter.ai/credits).

   > Note: this only lasts for the current terminal session. If you close
   > the terminal, you'll need to set it again next time.

## Running Each Task

```bash
python task1_faq_search.py       # FAQ knowledge base + keyword search demo
python task2_llm_integration.py  # LLM-rephrased FAQ answers demo
python task3_faq_matching.py     # keyword vs TF-IDF vs hybrid search comparison
python task4_helpdesk_agent.py   # scripted 4-scenario demo, then live interactive chat
```

## Task Summaries

### Task 1 — FAQ Knowledge Base Foundation
A small FAQ knowledge base (`FAQS`) with `id`, `category`, `question`,
`answer`, and `keywords` fields, plus:
- `search_by_keyword(faqs, query)` — case-insensitive, whole-word keyword
  matching, ranked by hit count.
- `get_faq_by_id(faqs, faq_id)`
- `get_faqs_by_category(faqs, category)`

### Task 2 — OpenRouter LLM Integration
`LLMClient` wraps the OpenRouter chat-completions API
(`openai/gpt-4o-mini` by default):
- `generate(prompt, system_message, max_tokens)` — raw chat completion
  call with descriptive error handling.
- `generate_faq_response(user_question, faq_entry)` — rephrases an FAQ's
  official answer conversationally, strictly grounded in the FAQ content
  (no hallucination), with an explicit fallback instruction if the FAQ
  doesn't fully answer the question.

### Task 3 — Intelligent FAQ Matching
`FAQMatcher` builds a TF-IDF index over each FAQ's question + keywords:
- `match(query, top_k)` — ranked `(faq, score)` results via cosine
  similarity.
- `best_match(query, threshold=0.15)` — single best match if confident
  enough, else `None`.
- `explain_match(query)` — human-readable top-3 breakdown.

`hybrid_search(faqs, query, top_k)` merges Task 1's keyword search
(flat 0.5 base score) with Task 3's TF-IDF scores, keeping the highest
score per FAQ.

### Task 4 — Complete Helpdesk Agent
`SupportAgent` orchestrates Tasks 1–3 into a stateful conversation:
- `handle_message(user_message)` — matches via `hybrid_search()`,
  generates a grounded response via `LLMClient`, or falls back with an
  escalation offer if nothing confident is found.
- `escalate(reason)` — creates a mock support ticket (`TICKET-XXXXX`).
- `get_conversation_summary()` — formatted transcript with FAQ IDs and
  confidence scores.
- `reset()` — clears conversation state.

An interactive chat interface supports free-text questions plus the
commands `history`, `escalate`, `reset`, and `quit`, and proactively
suggests escalation after 3 consecutive low-confidence answers.

## Notes

- If `OPENROUTER_API_KEY` isn't set (or the API call fails for any
  reason), Task 2/4 gracefully fall back to the FAQ's raw answer text
  instead of crashing — this is intentional error handling, not a bug.
- `__pycache__/` is excluded via `.gitignore` — it's an auto-generated
  Python bytecode cache, not part of the project source.