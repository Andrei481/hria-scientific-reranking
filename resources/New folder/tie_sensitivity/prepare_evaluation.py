"""Freeze identical LitSearch prompts and candidate pools for every condition."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from experiment import prompt_text, read_jsonl, sha256_file, source_hashes, write_json


def freeze_retriever(rows, corpus, output, expected_count=597, k=100):
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} queries, found {len(rows)}")
    seen = set()
    with output.open("x", encoding="utf-8") as handle:
        for row in rows:
            identity = [row["query"], row.get("query_set"), row.get("specificity")]
            qid = hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()
            if qid in seen:
                raise ValueError("Duplicate query identity")
            seen.add(qid)
            ids = [int(i) for i in row["retrieved"][:k]]
            if len(ids) != k or len(set(ids)) != k:
                raise ValueError(f"Need {k} unique first-stage candidates")
            docs = [corpus[i] for i in ids]
            if any(not d.strip() for d in docs):
                raise ValueError("Missing candidate text")
            record = {
                "qid": qid, "query": row["query"], "query_set": row.get("query_set"),
                "specificity": row.get("specificity"), "corpusids": [int(i) for i in row["corpusids"]],
                "pre_reranked": ids, "prompt": prompt_text(row["query"], docs),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-root", type=Path, required=True)
    parser.add_argument("--corpus-revision", required=True, help="Pinned LitSearch dataset commit SHA")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "evaluation_inputs")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.corpus_revision):
        raise ValueError("--corpus-revision must be a commit SHA")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use an empty output directory; frozen evaluation inputs are never overwritten")
    from datasets import load_dataset
    corpus = load_dataset("princeton-nlp/LitSearch", "corpus_clean", split="full", revision=args.corpus_revision)
    lookup = {int(r["corpusid"]): f"Title: {r['title']}\nAbstract: {r['abstract']}" for r in corpus}
    args.output.mkdir(parents=True, exist_ok=True)
    files = {}
    identity_sets = []
    for retriever in ("bm25", "grit", "qwen3"):
        source = args.retrieval_root / f"LitSearch.title_abstract.{retriever}.jsonl"
        output = args.output / f"{retriever}.jsonl"
        freeze_retriever(read_jsonl(source), lookup, output)
        identity_sets.append({r["qid"] for r in read_jsonl(output)})
        files[retriever] = {"input_sha256": sha256_file(output), "retrieval_sha256": sha256_file(source), "queries": 597}
        print(f"Frozen {retriever}: {output}", flush=True)
    if not identity_sets[0] == identity_sets[1] == identity_sets[2]:
        raise ValueError("Retrievers do not contain the same query identities")
    write_json(args.output / "manifest.json", {
        "dataset": "princeton-nlp/LitSearch", "revision": args.corpus_revision,
        "max_k": 100, "candidate_word_limit": 300, "files": files,
        "source_hashes": source_hashes(),
    })


if __name__ == "__main__":
    main()
