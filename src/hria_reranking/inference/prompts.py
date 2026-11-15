"""RankGPT-style listwise prompt construction."""

from __future__ import annotations

from collections.abc import Sequence


def format_listwise_prompt(query: str, passages: Sequence[str]) -> str:
    """Format the instruction used for training and full-list inference."""
    if not passages:
        raise ValueError("passages must not be empty")

    formatted_passages = []
    for index, passage in enumerate(passages, start=1):
        passage_text = " ".join(str(passage).split()[:300])
        formatted_passages.append(f"[{index}] {passage_text}")
    passage_block = "\n\n---\n\n".join(formatted_passages)

    return (
        "You are an expert academic paper reranker. "
        "Your task is to re-order the given list of passages "
        f"(from [1] to [{len(passages)}]) based on their relevance to the query. "
        "Respond with only the ranking and nothing else.\n\n"
        "Example output for 8 passages:\n"
        "[2] > [5] > [4] > [8] > [6] > [1] > [3] > [7]\n\n"
        f"Query: {query}\n\n"
        f"Passages:\n{passage_block}\n\n"
        "Your ranking (most to least relevant):"
    )
