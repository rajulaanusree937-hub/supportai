"""
Task 3 — Intelligent FAQ Matching
===================================

Task 1's keyword search only finds an FAQ if the user's words literally
appear in its keywords/question/category. That fails on paraphrases like
"I forgot my login credentials", which shares almost no exact words with
the password-reset FAQ despite meaning the same thing.

This module adds a smarter, meaning-aware matcher on top of Task 1:
    1. `FAQMatcher` — TF-IDF vectorisation + cosine similarity over each
       FAQ's question + keywords, so semantically related phrasing scores
       highly even without exact word overlap.
    2. `hybrid_search()` — combines Task 1's exact keyword search with
       TF-IDF matching, keeping the best score per FAQ.
    3. A comparison demo showing keyword search vs TF-IDF vs hybrid on
       three paraphrased queries.

This module REUSES Task 1's `FAQS` list and `search_by_keyword()` function
rather than redefining them.

Setup
-----
    pip install scikit-learn
"""

from typing import List, Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Reuse Task 1's knowledge base and keyword search instead of redefining them.
from task1_faq_search import FAQS, search_by_keyword


# ---------------------------------------------------------------------------
# 1. FAQMatcher — TF-IDF + cosine similarity
# ---------------------------------------------------------------------------

class FAQMatcher:
    """
    Semantic-ish FAQ matcher built on TF-IDF vectors and cosine similarity.

    Each FAQ is represented by a single "document" made of its question
    text plus its keyword list. TF-IDF weights words by how distinctive
    they are across the FAQ set, so common filler words matter less and
    topic-specific words (e.g. "password", "refund") matter more. Cosine
    similarity then measures how closely a user's query points in the same
    "direction" as each FAQ's document, even when the exact wording differs.
    """

    def __init__(self, faqs: List[Dict]):
        """
        Build the TF-IDF index over all FAQs.

        Args:
            faqs: list of FAQ dictionaries (as defined in Task 1), each
                with at least "question" and "keywords" fields.
        """
        self.faqs = faqs

        # One combined text document per FAQ: question + keywords.
        # Combining both gives the vectorizer more signal per FAQ than
        # the question text alone would.
        self.corpus = [
            f"{faq['question']} {' '.join(faq['keywords'])}" for faq in faqs
        ]

        # Fit a TF-IDF vectorizer over the FAQ corpus. English stopwords
        # (the, is, a, ...) are stripped since they carry no topical signal.
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.faq_vectors = self.vectorizer.fit_transform(self.corpus)

    def match(self, query: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """
        Rank all FAQs by cosine similarity to the query.

        Args:
            query: the user's free-text question.
            top_k: maximum number of results to return.

        Returns:
            A list of (faq_dict, confidence_score) tuples, sorted by score
            descending, limited to `top_k` entries. Scores are floats in
            [0.0, 1.0], rounded to 4 decimal places. FAQs with zero
            similarity are excluded.
        """
        # Project the query into the same TF-IDF space as the FAQ corpus.
        query_vector = self.vectorizer.transform([query])

        # Cosine similarity between the query and every FAQ vector.
        similarities = cosine_similarity(query_vector, self.faq_vectors)[0]

        # Pair each FAQ with its score, keep only non-zero matches.
        scored = [
            (self.faqs[i], round(float(score), 4))
            for i, score in enumerate(similarities)
            if score > 0
        ]

        # Highest similarity first.
        scored.sort(key=lambda pair: pair[1], reverse=True)

        return scored[:top_k]

    def best_match(
        self, query: str, threshold: float = 0.15
    ) -> Optional[Tuple[Dict, float]]:
        """
        Return the single best-matching FAQ, if confident enough.

        Args:
            query: the user's free-text question.
            threshold: minimum cosine similarity score required to accept
                a match (0.0-1.0). 0.15 works well for small FAQ sets.

        Returns:
            A (faq_dict, confidence_score) tuple for the top match if its
            score >= threshold, otherwise None.
        """
        top_matches = self.match(query, top_k=1)
        if not top_matches:
            return None

        best_faq, best_score = top_matches[0]
        if best_score >= threshold:
            return best_faq, best_score
        return None

    def explain_match(self, query: str) -> str:
        """
        Produce a human-readable breakdown of the top 3 matches for a query,
        useful for debugging/demoing why a particular FAQ was (or wasn't)
        selected.

        Args:
            query: the user's free-text question.

        Returns:
            A formatted multi-line string listing up to 3 ranked matches
            with their scores, or a "no matches" message if none score
            above zero.
        """
        top_matches = self.match(query, top_k=3)

        if not top_matches:
            return f"No matches found for query: '{query}'"

        lines = [f"Top matches for query: '{query}'"]
        for rank, (faq, score) in enumerate(top_matches, start=1):
            lines.append(f"  {rank}. [{score:.4f}] {faq['question']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Hybrid Search
# ---------------------------------------------------------------------------

def hybrid_search(
    faqs: List[Dict], query: str, top_k: int = 3
) -> List[Tuple[Dict, float]]:
    """
    Combine Task 1's exact keyword search with Task 3's TF-IDF matching
    into a single ranked result list.

    Scoring approach:
        - Every FAQ returned by `search_by_keyword()` gets a flat base
          score of 0.5 (a simple, interpretable signal that at least one
          exact keyword/question/category word matched).
        - Every FAQ returned by `FAQMatcher.match()` gets its cosine
          similarity score (0.0-1.0).
        - If an FAQ appears in both result sets, keep the HIGHER of the
          two scores rather than summing them, so a strong TF-IDF match
          isn't unfairly boosted just because it also had a keyword hit,
          and vice versa.
        - Final list is deduplicated by FAQ id, sorted by score descending,
          and truncated to `top_k`.

    Args:
        faqs: list of FAQ dictionaries to search.
        query: the user's free-text question.
        top_k: maximum number of results to return.

    Returns:
        A list of (faq_dict, score) tuples, sorted by score descending.
    """
    KEYWORD_BASE_SCORE = 0.5

    # Collect scores per FAQ id, tracking the best score seen so far.
    best_scores: Dict[str, float] = {}
    faq_by_id: Dict[str, Dict] = {faq["id"]: faq for faq in faqs}

    # --- Keyword search results (Task 1) ---
    for faq in search_by_keyword(faqs, query):
        best_scores[faq["id"]] = max(
            best_scores.get(faq["id"], 0.0), KEYWORD_BASE_SCORE
        )

    # --- TF-IDF results (Task 3) ---
    matcher = FAQMatcher(faqs)
    for faq, score in matcher.match(query, top_k=len(faqs)):
        best_scores[faq["id"]] = max(best_scores.get(faq["id"], 0.0), score)

    # Build final (faq, score) list, sorted by score descending.
    merged = [
        (faq_by_id[faq_id], round(score, 4)) for faq_id, score in best_scores.items()
    ]
    merged.sort(key=lambda pair: pair[1], reverse=True)

    return merged[:top_k]


# ---------------------------------------------------------------------------
# 3. Comparison Demonstration
# ---------------------------------------------------------------------------

def print_comparison(faqs: List[Dict], matcher: FAQMatcher, query: str) -> None:
    """
    Run the same query through keyword search, TF-IDF matching, and hybrid
    search, and print all three results side by side for comparison.

    Args:
        faqs: list of FAQ dictionaries to search.
        matcher: a pre-built FAQMatcher instance (reused across queries so
            the TF-IDF index isn't rebuilt every call).
        query: the query string to demonstrate.
    """
    print(f"Query: {query}")

    # --- Keyword search (Task 1) ---
    print("[Keyword Search]")
    keyword_results = search_by_keyword(faqs, query)
    if not keyword_results:
        print("  (no results)")
    else:
        for rank, faq in enumerate(keyword_results, start=1):
            print(f"  {rank}. {faq['question']}")

    # --- TF-IDF matching (Task 3) ---
    print("[TF-IDF Matching]")
    tfidf_results = matcher.match(query, top_k=3)
    if not tfidf_results:
        print("  (no results)")
    else:
        for rank, (faq, score) in enumerate(tfidf_results, start=1):
            print(f"  {rank}. [{score:.4f}] {faq['question']}")

    # --- Hybrid search (keyword + TF-IDF merged) ---
    print("[Hybrid Search]")
    hybrid_results = hybrid_search(faqs, query, top_k=3)
    if not hybrid_results:
        print("  (no results)")
    else:
        for rank, (faq, score) in enumerate(hybrid_results, start=1):
            print(f"  {rank}. [{score:.4f}] {faq['question']}")

    # --- Best single match, with threshold logic ---
    best = matcher.best_match(query)
    if best:
        best_faq, best_score = best
        print(f"Best match: {best_faq['question']} (confidence: {best_score:.4f})")
    else:
        print("Best match: none (below confidence threshold)")

    print()  # blank line between queries


if __name__ == "__main__":
    faq_matcher = FAQMatcher(FAQS)  # build the TF-IDF index once, reuse for all queries

    demo_queries = [
        "I forgot my login credentials",
        "Can I get my money back?",
        "package delivery time",
    ]

    for demo_query in demo_queries:
        print_comparison(FAQS, faq_matcher, demo_query)

    # Extra: explain_match() output for one query
    print(faq_matcher.explain_match("I forgot my login credentials"))