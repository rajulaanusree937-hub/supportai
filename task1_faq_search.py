"""
Task 1 — FAQ Knowledge Base Foundation
=======================================

This module builds the data foundation for SupportAI:
    1. A small FAQ knowledge base (list of dictionaries).
    2. Keyword-based search functions over that knowledge base.
    3. A demonstration that runs a few sample queries.

This is intentionally dependency-free (no external libraries, no LLM calls).
The `FAQS` list and the search functions defined here are meant to be
imported and reused directly in Tasks 2, 3 and 4 rather than rebuilt.
"""

from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# 1. FAQ Knowledge Base
# ---------------------------------------------------------------------------
# Each FAQ is a dictionary with a unique id, a category, the question text,
# the official answer, and a list of keywords that a user might type when
# searching for this topic. Keywords supplement the question/category text
# so that short or informal queries can still find the right entry.

FAQS: List[Dict] = [
    {
        "id": "faq-001",
        "category": "Account",
        "question": "How do I reset my password?",
        "answer": (
            "Click 'Forgot Password' on the login page. Enter your registered "
            "email address and check your inbox for a reset link valid for "
            "24 hours."
        ),
        "keywords": ["password", "reset", "forgot", "login", "account", "credentials"],
    },
    {
        "id": "faq-002",
        "category": "Billing",
        "question": "What is your refund policy?",
        "answer": (
            "We offer full refunds within 30 days of purchase for unused "
            "subscriptions. Partial refunds are available for annual plans "
            "cancelled after 30 days."
        ),
        "keywords": ["refund", "money back", "billing", "cancel", "payment", "policy"],
    },
    {
        "id": "faq-003",
        "category": "Billing",
        "question": "How do I update my payment method?",
        "answer": (
            "Go to Account Settings > Billing > Payment Methods, then click "
            "'Add New Card' or 'Edit' next to an existing card to update it."
        ),
        "keywords": ["payment", "card", "billing", "update", "credit card", "method"],
    },
    {
        "id": "faq-004",
        "category": "Technical",
        "question": "The app keeps crashing, what should I do?",
        "answer": (
            "Try restarting the app first. If that doesn't work, update to the "
            "latest version from your app store, then restart your device. "
            "Contact support if the issue persists."
        ),
        "keywords": ["crash", "crashing", "bug", "error", "technical", "app", "not working"],
    },
    {
        "id": "faq-005",
        "category": "Account",
        "question": "How do I delete my account?",
        "answer": (
            "Go to Account Settings > Privacy > Delete Account. Note that this "
            "action is permanent and will remove all your data after 30 days."
        ),
        "keywords": ["delete", "account", "close", "remove", "deactivate", "privacy"],
    },
    {
        "id": "faq-006",
        "category": "Shipping",
        "question": "How can I track my order?",
        "answer": (
            "Once your order ships, you'll receive a tracking number by email. "
            "You can also view tracking status under 'My Orders' in your account."
        ),
        "keywords": ["track", "tracking", "order", "shipping", "delivery", "status"],
    },
]


# ---------------------------------------------------------------------------
# 2. Search Functions
# ---------------------------------------------------------------------------

# Very common words that carry little search meaning on their own. Ignoring
# these stops trivial matches like "my" from padding out hit counts on
# unrelated FAQs (e.g. matching "my" inside "My Orders").
STOPWORDS = {"my", "the", "a", "an", "is", "are", "to", "for", "of", "on", "i"}


def search_by_keyword(faqs: List[Dict], query: str) -> List[Dict]:
    """
    Search FAQs by matching individual words in `query` against each FAQ's
    keywords, question, and category fields.

    Matching is case-insensitive and whole-word (so "pass" won't match
    "password"). Common stopwords (e.g. "my", "the") are ignored so they
    don't dilute results. FAQs are ordered by the number of query-word
    "hits" found (most matches first); FAQs with zero hits are excluded.

    Args:
        faqs: list of FAQ dictionaries to search.
        query: free-text search string, e.g. "forgot my password".

    Returns:
        A list of matching FAQ dictionaries, sorted by relevance
        (descending hit count). Empty list if nothing matches.
    """
    # Normalise the query into meaningful lowercase words (drop stopwords).
    query_words = [word for word in query.lower().split() if word not in STOPWORDS]

    scored_results = []  # list of (hit_count, faq) tuples

    for faq in faqs:
        # Build one combined, lowercase set of whole words per FAQ from its
        # keywords, question, and category, so matching is on whole words
        # rather than fragile substrings.
        searchable_words = set(" ".join(faq["keywords"]).lower().split())
        searchable_words |= set(faq["question"].lower().split())
        searchable_words |= set(faq["category"].lower().split())

        # Count how many of the query words appear as whole words in the FAQ.
        hit_count = sum(1 for word in query_words if word in searchable_words)

        if hit_count > 0:
            scored_results.append((hit_count, faq))

    # Sort by hit_count descending. Python's sort is stable, so FAQs with
    # equal hit counts keep their original relative order.
    scored_results.sort(key=lambda pair: pair[0], reverse=True)

    # Strip out the hit counts before returning, keeping only the FAQ dicts.
    return [faq for _, faq in scored_results]


def get_faq_by_id(faqs: List[Dict], faq_id: str) -> Optional[Dict]:
    """
    Look up a single FAQ by its unique id.

    Args:
        faqs: list of FAQ dictionaries to search.
        faq_id: the id to look for, e.g. "faq-001".

    Returns:
        The matching FAQ dictionary, or None if no FAQ has that id.
    """
    for faq in faqs:
        if faq["id"] == faq_id:
            return faq
    return None


def get_faqs_by_category(faqs: List[Dict], category: str) -> List[Dict]:
    """
    Return all FAQs belonging to a given category (case-insensitive).

    Args:
        faqs: list of FAQ dictionaries to search.
        category: category name to filter by, e.g. "Billing".

    Returns:
        A list of FAQ dictionaries whose category matches (case-insensitive).
        Empty list if no FAQs belong to that category.
    """
    target_category = category.lower()
    return [faq for faq in faqs if faq["category"].lower() == target_category]


# ---------------------------------------------------------------------------
# 3. Demonstration
# ---------------------------------------------------------------------------

def print_search_results(faqs: List[Dict], query: str) -> None:
    """
    Run search_by_keyword() for a given query and pretty-print the results
    in the format: [Category] Question -> Answer.

    Args:
        faqs: list of FAQ dictionaries to search.
        query: the search string to demonstrate.
    """
    print(f"Query: {query}")
    results = search_by_keyword(faqs, query)

    if not results:
        print("  No matching FAQs found.")
    else:
        for faq in results:
            print(f"  [{faq['category']}] {faq['question']}")
            print(f"  → {faq['answer']}")
    print()  # blank line between queries for readability


if __name__ == "__main__":
    # Sample queries covering: a clear match, a single-keyword match, and a
    # query with no relevant FAQs at all.
    demo_queries = [
        "forgot my password",
        "refund",
        "weather today",
    ]

    for demo_query in demo_queries:
        print_search_results(FAQS, demo_query)

    # Quick demonstration of the other two lookup functions.
    print("get_faq_by_id('faq-002'):")
    print(" ", get_faq_by_id(FAQS, "faq-002"))
    print()

    print("get_faqs_by_category('billing'):")
    for faq in get_faqs_by_category(FAQS, "billing"):
        print(f"  [{faq['category']}] {faq['question']}")