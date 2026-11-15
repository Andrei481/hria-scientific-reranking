import subprocess

# List of commands to execute sequentially

output_dir_model = "Qwen3-Reranker-8B"
commands = [
    f"python -m eval.reranking.rerank --retrieval_results_file results/retrieval/LitSearch.title_abstract.bm25.jsonl --use_local_llm --local_llm_url http://127.0.0.1:8000 --max_k 100 --output_dir results/{output_dir_model}/k_100",
    # f"python -m eval.reranking.rerank --retrieval_results_file results/retrieval/LitSearch.title_abstract.gtr.jsonl --use_local_llm --local_llm_url http://127.0.0.1:8000 --max_k 100 --output_dir results/{output_dir_model}/k_100",
    f"python -m eval.reranking.rerank --retrieval_results_file results/retrieval/LitSearch.title_abstract.grit.jsonl --use_local_llm --local_llm_url http://127.0.0.1:8000 --max_k 100 --output_dir results/{output_dir_model}/k_100",
    f"python -m eval.reranking.rerank --retrieval_results_file results/retrieval/LitSearch.title_abstract.qwen3.jsonl --use_local_llm --local_llm_url http://127.0.0.1:8000 --max_k 100 --output_dir results/{output_dir_model}/k_100",

    f"python -m eval.reranking.rerank --retrieval_results_file results/retrieval/LitSearch.title_abstract.bm25.jsonl --use_local_llm --local_llm_url http://127.0.0.1:8000 --max_k 20 --output_dir results/{output_dir_model}/k_20",
    # f"python -m eval.reranking.rerank --retrieval_results_file results/retrieval/LitSearch.title_abstract.gtr.jsonl --use_local_llm --local_llm_url http://127.0.0.1:8000 --max_k 20 --output_dir results/{output_dir_model}/k_20",
    f"python -m eval.reranking.rerank --retrieval_results_file results/retrieval/LitSearch.title_abstract.grit.jsonl --use_local_llm --local_llm_url http://127.0.0.1:8000 --max_k 20 --output_dir results/{output_dir_model}/k_20",
    f"python -m eval.reranking.rerank --retrieval_results_file results/retrieval/LitSearch.title_abstract.qwen3.jsonl --use_local_llm --local_llm_url http://127.0.0.1:8000 --max_k 20 --output_dir results/{output_dir_model}/k_20"


]

for cmd in commands:
    print(f"\n🚀 Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    
    # Optional: stop execution if one command fails
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        break
    else:
        print(f"✅ Finished: {cmd}\n")
