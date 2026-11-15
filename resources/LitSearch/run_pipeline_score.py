import subprocess
import sys


OUTPUT_DIR_MODEL = "Qwen3-Reranker-8B-score"
MODEL = "Qwen/Qwen3-Reranker-8B"
LOCAL_LLM_URL = "http://127.0.0.1:8000"
RERANK_ENDPOINT = "/v1/rerank"

RETRIEVAL_RESULT_FILES = [
    "results/retrieval/LitSearch.title_abstract.bm25.jsonl",
    # "results/retrieval/LitSearch.title_abstract.gtr.jsonl",
    "results/retrieval/LitSearch.title_abstract.grit.jsonl",
    "results/retrieval/LitSearch.title_abstract.qwen3.jsonl",
]


def build_command(retrieval_results_file: str, max_k: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "eval.reranking.rerank_score",
        "--retrieval_results_file",
        retrieval_results_file,
        "--model",
        MODEL,
        "--local_llm_url",
        LOCAL_LLM_URL,
        "--rerank_endpoint",
        RERANK_ENDPOINT,
        "--max_k",
        str(max_k),
        "--output_dir",
        f"results/{OUTPUT_DIR_MODEL}/k_{max_k}",
    ]


commands = [
    build_command(retrieval_results_file, max_k)
    for max_k in (100, 20)
    for retrieval_results_file in RETRIEVAL_RESULT_FILES
]


def main() -> None:
    for command in commands:
        print(f"\nRunning: {' '.join(command)}\n")
        result = subprocess.run(command)

        if result.returncode != 0:
            print(f"Command failed with exit code {result.returncode}: {' '.join(command)}")
            break

        print(f"Finished: {' '.join(command)}\n")


if __name__ == "__main__":
    main()
