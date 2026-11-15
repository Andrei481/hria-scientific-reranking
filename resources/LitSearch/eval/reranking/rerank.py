import os
import copy
import json
import argparse
import logging
import datasets
from tqdm import tqdm
from typing import List, Tuple
from utils import utils
from utils.openai_utils import OPENAIBaseEngine
import re
from datetime import datetime

# ---------------------------
# Logging configuration
# ---------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join("logs", timestamp)
LOG_FILENAME = "reranker_debug.log"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, LOG_FILENAME)

logging.basicConfig(
    filename=log_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# ---------------------------
# QUERY CONSTRUCTION FUNCTIONS
# ---------------------------

def create_prompt_messages(item: dict, rank_start: int, rank_end: int, index_type: str) -> List[dict]:
    """
    Build the chat-format messages for the LLM, and log the full constructed prompt.
    """
    query = item["query"]
    docs = item["documents"][rank_start:rank_end]
    num_docs = len(docs)

    if index_type == "title_abstract":
        max_words = 300
    elif index_type == "full_paper":
        max_words = 10000
    else:
        raise ValueError(f"Invalid index type: {index_type}")

    passages_str_list = []

    # Start enumeration at 1 to match training logic [1], [2], etc.
    for i, document in enumerate(docs, start=1):
        raw_content = document.get("content", "").strip()
        content = " ".join(raw_content.split()[:max_words])
        passages_str_list.append(f"[{i}] {content}")

    # Use the separator that matches training
    passages_for_prompt = "\n\n---\n\n".join(passages_str_list)

    user_message = (
        f"You are an expert academic paper reranker. "
        f"Your task is to re-order the given list of passages (from [1] to [{num_docs}]) "
        f"based on their relevance to the query. Respond with only the ranking and nothing else.\n\n"
        f"Example output for 8 passages:\n"
        f"[2] > [5] > [4] > [8] > [6] > [1] > [3] > [7]\n\n"
        f"Query: {query}\n\n"
        f"Passages:\n{passages_for_prompt}\n\n"
        f"Your ranking (most to least relevant):"
    )

    # repeated_user_message = user_message + "\n\n" + user_message

    messages = [{"role": "user", "content": user_message}]
    # messages = [{"role": "user", "content": repeated_user_message}]

    # Log the constructed prompt
    try:
        logger.debug(
            "Constructed prompt for query '%s' (docs %d..%d -> %d docs):\n%s",
            query, rank_start, rank_end - 1, num_docs, user_message
        )
    except Exception as e:
        logger.debug(
            "Constructed prompt for query '%s' (docs %d..%d -> %d docs). "
            "Prompt not shown due to logging error: %s",
            query, rank_start, rank_end - 1, num_docs, str(e)
        )

    return messages

# ---------------------------
# RESPONSE PROCESSING FUNCTIONS
# ---------------------------

def clean_response(response: str):
    """
    Keep only digits and spaces. Return trimmed string.
    """
    new_response = ""
    for c in response:
        new_response += (c if c.isdigit() else " ")
    return new_response.strip()

def remove_duplicate(response: List[int]):
    """
    Remove duplicates, preserving order.
    """
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

def receive_permutation(item, permutation, rank_start, rank_end):
    """
    Robustly parses the LLM output for document indices like [1], [2], etc.
    Ignores bullet points (1., 2.) and other noise.
    """
    if permutation is None:
        logger.warning("receive_permutation called with None permutation.")
        return item

    # Extract bracketed indices only: [1], [10], etc.
    matches = re.findall(r"\[(\d+)\]", str(permutation))

    parsed = []
    for m in matches:
        try:
            parsed.append(int(m) - 1)
        except ValueError:
            pass

    logger.debug("Regex parsed indices: %s", parsed)

    parsed_unique = []
    seen = set()
    original_count = rank_end - rank_start

    for p in parsed:
        if p not in seen and 0 <= p < original_count:
            parsed_unique.append(p)
            seen.add(p)

    original_rank = list(range(original_count))
    final_order = parsed_unique + [i for i in original_rank if i not in parsed_unique]

    logger.debug("Final applied order: %s", final_order)

    # Apply permutation
    cut_range = copy.deepcopy(item["documents"][rank_start:rank_end])

    for j, x in enumerate(final_order):
        item["documents"][j + rank_start] = copy.deepcopy(cut_range[x])

        # Restore metadata if present
        if "rank" in item["documents"][j + rank_start]:
            try:
                item["documents"][j + rank_start]["rank"] = cut_range[j].get("rank")
            except Exception:
                pass

    return item

# ---------------------------
# PERMUTATION PIPELINE
# ---------------------------

def permutation_pipeline(model, item: dict, rank_start: int, rank_end: int, index_type: str) -> dict:
    window_size = rank_end - rank_start
    decrement_rate = max(1, window_size // 5)
    min_count = max(1, window_size // 2)

    attempt = 0
    max_attempts = 10

    while (rank_end - rank_start) >= min_count and attempt < max_attempts:
        try:
            attempt += 1
            logger.debug(
                "Permutation pipeline attempt %d for query '%s' with doc window [%d:%d] (size=%d)",
                attempt, item.get("query", "<no-query>"), rank_start, rank_end, rank_end - rank_start
            )

            messages = create_prompt_messages(item, rank_start, rank_end, index_type)

            try:
                permutation = model.generate(messages=messages)
            except TypeError:
                permutation = model.generate(messages)

            logger.debug(
                "Raw LLM output (type=%s):\n%s",
                type(permutation).__name__, str(permutation)
            )

            return receive_permutation(item, permutation, rank_start, rank_end)

        except Exception as e:
            old_window = rank_end - rank_start
            rank_end = max(rank_start + min_count, rank_end - decrement_rate)
            logger.exception(
                "Exception during permutation pipeline: %s. Reduced window from %d to %d and retrying.",
                str(e), old_window, rank_end - rank_start
            )

    logger.error(
        "Unable to rerank documents for query '%s' after %d attempts. Returning original order.",
        item.get("query", "<no-query>"), attempt
    )
    return item

# ---------------------------
# MAIN SCRIPT
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_file", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--max_k", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/reranking/")
    parser.add_argument("--dataset_path", default="princeton-nlp/LitSearch")
    parser.add_argument("--use_local_llm", action="store_true")
    parser.add_argument("--local_llm_url", default="http://127.0.0.1:8080")
    args = parser.parse_args()

    logger.info("Starting reranker with args: %s", vars(args))

    if args.use_local_llm:
        from utils.local_llm_utils import LocalLLMEngine
        model = LocalLLMEngine(api_url=args.local_llm_url)
    else:
        model = utils.get_gpt4_model(args.model, azure=True)

    corpus_data = datasets.load_dataset(args.dataset_path, "corpus_clean", split="full")
    retrieval_results = utils.read_json(args.retrieval_results_file)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        os.path.basename(args.retrieval_results_file).replace(".json", ".reranked.json")
    )

    index_type = os.path.basename(args.retrieval_results_file).split(".")[1]

    if index_type == "title_abstract":
        corpusid_to_text = {
            utils.get_clean_corpusid(item): utils.get_clean_title_abstract(item)
            for item in corpus_data
        }
    elif index_type == "full_paper":
        corpusid_to_text = {
            utils.get_clean_corpusid(item): utils.get_clean_full_paper(item)
            for item in corpus_data
        }
    else:
        raise ValueError(f"Invalid index type: {index_type}")

    for result in retrieval_results:
        result["retrieved"] = result.get("retrieved", [])[:args.max_k]

    reranking_inputs = []
    for query_info in retrieval_results:
        reranking_inputs.append({
            "query": query_info.get("query", ""),
            "documents": [
                {"content": corpusid_to_text.get(cid, ""), "corpusid": cid}
                for cid in query_info.get("retrieved", [])
            ]
        })

    if not os.path.exists(output_file):
        utils.write_json(copy.deepcopy(retrieval_results), output_file)

    for item_idx, item in enumerate(tqdm(reranking_inputs)):
        try:
            reranking_outputs = utils.read_json(output_file)

            if "pre_reranked" in reranking_outputs[item_idx]:
                continue

            reranked_item = permutation_pipeline(
                model, item, 0, len(item["documents"]), index_type
            )

            reranking_outputs[item_idx]["pre_reranked"] = reranking_outputs[item_idx]["retrieved"]
            reranking_outputs[item_idx]["retrieved"] = [
                doc.get("corpusid") for doc in reranked_item.get("documents", [])
            ]

            utils.write_json(reranking_outputs, output_file, silent=True)

        except Exception as e:
            logger.exception(
                "Unhandled exception while processing query idx %d: %s",
                item_idx, str(e)
            )

    logger.info("Reranking run completed. Logs written to: %s", log_path)
