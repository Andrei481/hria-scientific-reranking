import argparse
import copy
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from tqdm import tqdm


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join("logs", timestamp)
LOG_FILENAME = "reranker_rankzephyr_debug.log"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, LOG_FILENAME)

logging.basicConfig(
    filename=log_path,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def configure_cuda_multiprocessing() -> str:
    """Use spawn multiprocessing so vLLM can initialize CUDA in child processes."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
    start_method = mp.get_start_method(allow_none=False)
    logger.info(
        "Configured multiprocessing start method=%s, VLLM_WORKER_MULTIPROC_METHOD=%s",
        start_method,
        os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
    )
    return start_method


def import_rankllm_data():
    try:
        from rank_llm import data as rankllm_data
    except ImportError as exc:
        raise ImportError(
            "RankLLM is required for this runner. Install it in a separate "
            "environment, for example: pip install 'rank-llm[vllm]'"
        ) from exc
    return rankllm_data


def patch_transformers_pruning_helpers() -> List[str]:
    """Patch a RankLLM/Transformers import compatibility issue.

    RankLLM 0.25.x still imports pruning helpers from transformers.modeling_utils.
    Newer Transformers versions moved those helpers to transformers.pytorch_utils.
    """
    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError:
        return []

    try:
        import transformers.pytorch_utils as pytorch_utils
    except ImportError:
        logger.debug(
            "Could not import transformers.pytorch_utils for RankLLM compatibility patch",
            exc_info=True,
        )
        return []

    helper_names = [
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
        "prune_conv1d_layer",
        "prune_layer",
    ]
    patched = []
    for helper_name in helper_names:
        if hasattr(modeling_utils, helper_name):
            continue
        if not hasattr(pytorch_utils, helper_name):
            logger.debug("transformers.pytorch_utils is missing %s", helper_name)
            continue
        setattr(modeling_utils, helper_name, getattr(pytorch_utils, helper_name))
        patched.append(helper_name)

    if patched:
        logger.info(
            "Patched transformers.modeling_utils helpers from transformers.pytorch_utils: %s",
            ", ".join(patched),
        )
    return patched


def import_rankzephyr_classes():
    patch_transformers_pruning_helpers()
    try:
        from rank_llm.rerank.listwise import RankListwiseOSLLM, ZephyrReranker
    except ImportError as exc:
        raise ImportError(
            "Could not import RankLLM listwise rerankers. If RankLLM is installed, "
            "this is usually a RankLLM/Transformers compatibility issue; this runner "
            "patches the known Transformers pruning-helper moves, but another "
            "dependency mismatch may remain. Check the chained traceback above."
        ) from exc
    return RankListwiseOSLLM, ZephyrReranker


def build_rankllm_request(
    item: Dict[str, Any],
    qid: str,
    data_module: Optional[Any] = None,
):
    """Convert one LitSearch reranking item into a RankLLM Request."""
    rankllm_data = data_module or import_rankllm_data()
    query = rankllm_data.Query(text=item.get("query", ""), qid=str(qid))
    candidates = []

    for document in item.get("documents", []):
        corpusid = str(document.get("corpusid"))
        content = str(document.get("content", "")).strip()
        candidates.append(
            rankllm_data.Candidate(
                docid=corpusid,
                score=0.0,
                doc={"text": content},
            )
        )

    return rankllm_data.Request(query=query, candidates=candidates)


def max_words_for_index_type(index_type: str) -> int:
    if index_type == "title_abstract":
        return 300
    if index_type == "full_paper":
        return 10000
    raise ValueError(f"Invalid index type: {index_type}")


def truncate_words(text: str, max_words: int) -> str:
    return " ".join(str(text or "").split()[:max_words])


def build_rankzephyr_documents(
    retrieved_ids: List[int],
    corpusid_to_text: Dict[int, str],
    index_type: str,
) -> List[Dict[str, Any]]:
    max_words = max_words_for_index_type(index_type)
    return [
        {
            "content": truncate_words(corpusid_to_text.get(cid, ""), max_words),
            "corpusid": cid,
        }
        for cid in retrieved_ids
    ]


def result_to_retrieved_order(result: Any) -> List[int]:
    """Return reranked corpus IDs from a RankLLM Result."""
    retrieved = []
    for candidate in getattr(result, "candidates", []):
        docid = candidate.docid
        try:
            retrieved.append(int(docid))
        except (TypeError, ValueError):
            retrieved.append(docid)
    return retrieved


def create_rankzephyr_reranker(
    model_path: str,
    context_size: int,
    window_size: int,
    stride: int,
    device: str,
    num_gpus: int,
    variable_passages: bool,
    base_url: Optional[str],
):
    """Create the RankZephyr reranker using RankLLM's official prompt/templates."""
    RankListwiseOSLLM, ZephyrReranker = import_rankzephyr_classes()

    if base_url:
        logger.info("Using RankLLM with external OpenAI-compatible vLLM server: %s", base_url)
        return RankListwiseOSLLM(
            model=model_path,
            name="rank_zephyr",
            context_size=context_size,
            device=device,
            num_gpus=num_gpus,
            variable_passages=variable_passages,
            window_size=window_size,
            stride=stride,
            base_url=base_url,
        )

    logger.info("Using RankLLM ZephyrReranker in-process with model: %s", model_path)
    return ZephyrReranker(
        model_path=model_path,
        context_size=context_size,
        device=device,
        num_gpus=num_gpus,
        variable_passages=variable_passages,
        window_size=window_size,
    )


def run_rankzephyr_rerank(args: argparse.Namespace) -> str:
    from utils import utils
    import datasets

    corpus_data = datasets.load_dataset(args.dataset_path, "corpus_clean", split="full")
    retrieval_results = utils.read_json(args.retrieval_results_file)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        os.path.basename(args.retrieval_results_file).replace(".json", ".reranked.json"),
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
        result["retrieved"] = result.get("retrieved", [])[: args.max_k]

    if not os.path.exists(output_file):
        utils.write_json(copy.deepcopy(retrieval_results), output_file)

    reranker = create_rankzephyr_reranker(
        model_path=args.model_path,
        context_size=args.context_size,
        window_size=args.window_size,
        stride=args.stride,
        device=args.device,
        num_gpus=args.num_gpus,
        variable_passages=args.variable_passages,
        base_url=args.base_url,
    )

    for item_idx, query_info in enumerate(tqdm(retrieval_results, desc="RankZephyr reranking")):
        reranking_outputs = utils.read_json(output_file, silent=True)
        if "pre_reranked" in reranking_outputs[item_idx]:
            continue

        item = {
            "query": query_info.get("query", ""),
            "documents": build_rankzephyr_documents(
                retrieved_ids=query_info.get("retrieved", []),
                corpusid_to_text=corpusid_to_text,
                index_type=index_type,
            ),
        }
        request = build_rankllm_request(item=item, qid=str(item_idx))
        result = reranker.rerank(
            request,
            rank_start=0,
            rank_end=len(item["documents"]),
            shuffle_candidates=False,
            logging=False,
            populate_invocations_history=args.populate_invocations_history,
            window_size=args.window_size,
            stride=args.stride,
            top_k_retrieve=len(item["documents"]),
        )

        if isinstance(result, list):
            result = result[0]

        reranking_outputs[item_idx]["pre_reranked"] = reranking_outputs[item_idx]["retrieved"]
        reranking_outputs[item_idx]["retrieved"] = result_to_retrieved_order(result)
        utils.write_json(reranking_outputs, output_file, silent=True)

    logger.info("RankZephyr reranking completed. Results saved to: %s", output_file)
    logger.info("Logs written to: %s", log_path)
    return output_file


def main() -> None:
    configure_cuda_multiprocessing()

    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_file", type=str, required=True)
    parser.add_argument("--max_k", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/rankzephyr/k_100")
    parser.add_argument("--dataset_path", default="princeton-nlp/LitSearch")
    parser.add_argument("--model_path", default="castorini/rank_zephyr_7b_v1_full")
    parser.add_argument("--context_size", type=int, default=4096)
    parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--variable_passages", action="store_true", default=True)
    parser.add_argument(
        "--base_url",
        default=None,
        help=(
            "Optional OpenAI-compatible vLLM base URL. If omitted, RankLLM "
            "loads RankZephyr in-process."
        ),
    )
    parser.add_argument("--populate_invocations_history", action="store_true")
    args = parser.parse_args()

    logger.info("Starting RankZephyr reranker with args: %s", vars(args))
    output_file = run_rankzephyr_rerank(args)
    print(f"Saved reranked results: {output_file}")
    print(f"Log file: {log_path}")


if __name__ == "__main__":
    main()
