import argparse
import copy
import json
import logging
import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple


LOG_FILENAME = "reranker_score_debug.log"
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
log_path = None


ScoreList = List[Tuple[int, float]]


def configure_logging(log_root: str = "logs") -> str:
    global log_path

    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_root, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, LOG_FILENAME)

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.handlers = [file_handler]
    logger.setLevel(logging.DEBUG)
    return log_path


def parse_rerank_response(payload: Any) -> ScoreList:
    """
    Parse common /v1/rerank response shapes into (document_index, score) pairs.
    Qwen reranker servers commonly return {"results": [{"index": ..., "relevance_score": ...}]}.
    """
    if isinstance(payload, dict):
        if isinstance(payload.get("scores"), list):
            return [(idx, float(score)) for idx, score in enumerate(payload["scores"])]

        candidates = payload.get("results") or payload.get("data") or payload.get("rankings")
    elif isinstance(payload, list):
        candidates = payload
    else:
        candidates = None

    if not isinstance(candidates, list):
        raise ValueError(f"Unsupported rerank response shape: {json.dumps(payload)[:500]}")

    parsed: ScoreList = []
    for fallback_idx, item in enumerate(candidates):
        if isinstance(item, dict):
            index = (
                item.get("index")
                if item.get("index") is not None
                else item.get("document_index", item.get("id", fallback_idx))
            )
            score = (
                item.get("relevance_score")
                if item.get("relevance_score") is not None
                else item.get("score", item.get("logit"))
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            index, score = item[0], item[1]
        else:
            continue

        if score is None:
            continue
        parsed.append((int(index), float(score)))

    if not parsed:
        raise ValueError(f"No scores found in rerank response: {json.dumps(payload)[:500]}")

    return parsed


def call_rerank_endpoint(
    query: str,
    documents: Sequence[str],
    model_name: str,
    api_url: str,
    endpoint: str,
    session: Optional[Any] = None,
    timeout: int = 300,
) -> ScoreList:
    if session is None:
        import requests

        session = requests

    url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
    payload = {
        "model": model_name,
        "query": query,
        "documents": list(documents),
        "return_documents": False,
    }

    logger.debug("Calling rerank endpoint %s with %d documents", url, len(documents))
    response = session.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return parse_rerank_response(response.json())


def apply_scores_to_result(result: dict, scores: Iterable[Tuple[int, float]], max_k: int) -> dict:
    updated = copy.deepcopy(result)
    original_retrieved = list(updated.get("retrieved", []))[:max_k]
    updated["pre_reranked"] = list(original_retrieved)

    score_by_index = {}
    for index, score in scores:
        if 0 <= index < len(original_retrieved) and index not in score_by_index:
            score_by_index[index] = float(score)

    scored_indices = sorted(score_by_index, key=lambda idx: (-score_by_index[idx], idx))
    missing_indices = [idx for idx in range(len(original_retrieved)) if idx not in score_by_index]
    final_indices = scored_indices + missing_indices

    updated["retrieved"] = [original_retrieved[idx] for idx in final_indices]
    return updated


def truncate_words(text: str, max_words: int) -> str:
    return " ".join(str(text or "").split()[:max_words])


def max_words_for_index_type(index_type: str) -> int:
    if index_type == "title_abstract":
        return 300
    if index_type == "full_paper":
        return 10000
    raise ValueError(f"Invalid index type: {index_type}")


def build_rerank_documents(
    retrieved_ids: Sequence[int],
    corpusid_to_text: dict,
    index_type: str,
) -> List[str]:
    max_words = max_words_for_index_type(index_type)
    return [truncate_words(corpusid_to_text.get(cid, ""), max_words) for cid in retrieved_ids]


def output_path_for(retrieval_results_file: str, output_dir: str) -> str:
    return os.path.join(
        output_dir,
        os.path.basename(retrieval_results_file).replace(".json", ".reranked.json"),
    )


def index_type_from_retrieval_file(retrieval_results_file: str) -> str:
    parts = os.path.basename(retrieval_results_file).split(".")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer index type from file name: {retrieval_results_file}")
    return parts[1]


def build_corpus_lookup(corpus_data: Any, index_type: str) -> dict:
    from utils import utils

    if index_type == "title_abstract":
        return {
            utils.get_clean_corpusid(item): utils.get_clean_title_abstract(item)
            for item in corpus_data
        }
    if index_type == "full_paper":
        return {
            utils.get_clean_corpusid(item): utils.get_clean_full_paper(item)
            for item in corpus_data
        }
    raise ValueError(f"Invalid index type: {index_type}")


def rerank_file(args: argparse.Namespace) -> str:
    import datasets
    from tqdm import tqdm
    from utils import utils

    current_log_path = configure_logging()
    logger.info("Starting score reranker with args: %s", vars(args))

    corpus_data = datasets.load_dataset(args.dataset_path, "corpus_clean", split="full")
    retrieval_results = utils.read_json(args.retrieval_results_file)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = output_path_for(args.retrieval_results_file, args.output_dir)
    index_type = index_type_from_retrieval_file(args.retrieval_results_file)
    corpusid_to_text = build_corpus_lookup(corpus_data, index_type)

    for result in retrieval_results:
        result["retrieved"] = result.get("retrieved", [])[: args.max_k]

    if not os.path.exists(output_file):
        utils.write_json(copy.deepcopy(retrieval_results), output_file)

    reranking_outputs = utils.read_json(output_file)
    for item_idx, result in enumerate(tqdm(reranking_outputs)):
        if "pre_reranked" in result:
            continue

        try:
            retrieved_ids = result.get("retrieved", [])[: args.max_k]
            documents = build_rerank_documents(retrieved_ids, corpusid_to_text, index_type)
            scores = call_rerank_endpoint(
                query=result.get("query", ""),
                documents=documents,
                model_name=args.model,
                api_url=args.local_llm_url,
                endpoint=args.rerank_endpoint,
                timeout=args.timeout,
            )
            reranking_outputs[item_idx] = apply_scores_to_result(result, scores, args.max_k)
            utils.write_json(reranking_outputs, output_file, silent=True)
        except Exception as exc:
            logger.exception(
                "Unhandled exception while processing query idx %d: %s",
                item_idx,
                str(exc),
            )

    logger.info("Score reranking completed. Logs written to: %s", current_log_path)
    return output_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--max_k", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/reranking/")
    parser.add_argument("--dataset_path", default="princeton-nlp/LitSearch")
    parser.add_argument("--local_llm_url", default="http://127.0.0.1:8000")
    parser.add_argument("--rerank_endpoint", default="/v1/rerank")
    parser.add_argument("--timeout", type=int, default=300)
    return parser.parse_args()


if __name__ == "__main__":
    rerank_file(parse_args())
