"""DORIS-MAE query decomposition used by the training notebooks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _required_collection(records: Mapping[str, Any], name: str) -> Any:
    if name not in records:
        raise KeyError(f"DORIS-MAE input is missing {name!r}")
    return records[name]


def _aspect_ids(query: Mapping[str, Any]) -> list[str]:
    """Return main and sub-aspect IDs once, in their source encounter order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw_main_id, raw_sub_ids in query["aspects"].items():
        for raw_aspect_id in (raw_main_id, *raw_sub_ids):
            aspect_id = str(raw_aspect_id)
            if aspect_id not in seen:
                seen.add(aspect_id)
                ordered.append(aspect_id)
    return ordered


def decompose_queries(records: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a DORIS-MAE bundle into one candidate-scored row per aspect.

    ``records`` is the original dataset object containing ``Query``,
    ``Annotation``, and ``aspect_id2aspect``.  The transformation mirrors the
    canonical ``dataset_creation.ipynb``: main-aspect text is assembled from
    ``aspect_id2sent``; sub-aspect text comes from the global aspect mapping;
    absent annotations receive score zero.

    The canonical release produces 1,501 rows.  Filtering the returned rows
    with ``any(score > 0 for score in row["candidate_scores"])`` retains the
    1,498 positive rows used by the training notebook.
    """
    if not isinstance(records, Mapping):
        raise TypeError("records must be a DORIS-MAE mapping")

    queries = _required_collection(records, "Query")
    annotations = _required_collection(records, "Annotation")
    aspect_text = _required_collection(records, "aspect_id2aspect")
    if not isinstance(queries, Sequence) or isinstance(queries, (str, bytes)):
        raise TypeError("records['Query'] must be a sequence")
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)):
        raise TypeError("records['Annotation'] must be a sequence")
    if not isinstance(aspect_text, Mapping):
        raise TypeError("records['aspect_id2aspect'] must be a mapping")

    annotation_lookup = {
        (int(annotation["abstract_id"]), str(annotation["aspect_id"])): int(
            annotation["score"]
        )
        for annotation in annotations
    }
    global_aspect_text = {str(key): value for key, value in aspect_text.items()}

    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        candidate_pool = [int(abstract_id) for abstract_id in query["candidate_pool"]]
        main_aspect_ids = {str(aspect_id) for aspect_id in query["aspects"]}
        main_text = {
            str(aspect_id): ". ".join(str(sentence) for sentence in sentences)
            for aspect_id, sentences in query.get("aspect_id2sent", {}).items()
            if sentences
        }

        for aspect_id in _aspect_ids(query):
            query_text = (
                main_text.get(aspect_id)
                if aspect_id in main_aspect_ids
                else global_aspect_text.get(aspect_id)
            )
            if query_text is None:
                continue

            candidate_scores = [
                annotation_lookup.get((abstract_id, aspect_id), 0)
                for abstract_id in candidate_pool
            ]
            rows.append(
                {
                    "row_id": len(rows),
                    "query_text": str(query_text),
                    "aspect_id": aspect_id,
                    "original_query_id": query_index,
                    "original_query_type": query["query_type"],
                    "candidate_pool": candidate_pool.copy(),
                    "candidate_scores": candidate_scores,
                }
            )

    return rows


def aggregate_full_queries(records: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build the notebook's no-aspect-decomposition ablation rows."""
    queries = _required_collection(records, "Query")
    annotations = _required_collection(records, "Annotation")
    annotation_lookup = {
        (int(annotation["abstract_id"]), str(annotation["aspect_id"])): int(
            annotation["score"]
        )
        for annotation in annotations
    }

    rows: list[dict[str, Any]] = []
    for query_index, query in enumerate(queries):
        candidate_pool = [int(abstract_id) for abstract_id in query["candidate_pool"]]
        aspect_ids = _aspect_ids(query)
        candidate_scores = [
            max(
                (
                    annotation_lookup.get((abstract_id, aspect_id), 0)
                    for aspect_id in aspect_ids
                ),
                default=0,
            )
            for abstract_id in candidate_pool
        ]
        rows.append(
            {
                "row_id": len(rows),
                "query_text": str(query["query_text"]),
                "aspect_id": "original_query",
                "original_query_id": query_index,
                "original_query_type": query["query_type"],
                "candidate_pool": candidate_pool,
                "candidate_scores": candidate_scores,
            }
        )
    return rows
