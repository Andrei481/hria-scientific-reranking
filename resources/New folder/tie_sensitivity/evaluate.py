"""Evaluate one served checkpoint, saving raw replies and pre-repair validity."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from experiment import litsearch_metrics, parse_ranking, read_jsonl, sha256_file, source_hashes, write_json


def complete_query(session, url, model, row, max_tokens):
    response = session.post(url + "/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": row["prompt"]}],
        "temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": max_tokens,
        "stream": False,
    }, timeout=600)
    response.raise_for_status()
    payload = response.json()
    choice = payload["choices"][0]
    text = choice["message"]["content"]
    if not isinstance(text, str):
        raise ValueError("Endpoint did not return string content")
    order, audit = parse_ranking(text, len(row["pre_reranked"]))
    retrieved = [row["pre_reranked"][i] for i in order]
    return {
        **{k: v for k, v in row.items() if k != "prompt"},
        "retrieved": retrieved, "raw_response": text, "parser": audit,
        "finish_reason": choice.get("finish_reason"), "usage": payload.get("usage"),
        "metrics": litsearch_metrics(retrieved, row["corpusids"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).parent / "evaluation_inputs")
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0, help="Smoke evaluation only; saved separately and excluded from analysis")
    args = parser.parse_args()
    if args.limit < 0 or args.max_tokens < 1:
        raise ValueError("Invalid limit or token budget")
    training_path = args.run / "training_manifest.json"
    training = json.loads(training_path.read_text())
    if training["status"] != "complete" and not args.limit:
        raise ValueError("Smoke checkpoint may only be evaluated with --limit")
    inputs_manifest_path = args.inputs / "manifest.json"
    inputs_manifest = json.loads(inputs_manifest_path.read_text())
    output = args.run / ("evaluation_SMOKE" if args.limit else "evaluation")
    protocol = {
        "training_manifest_sha256": sha256_file(training_path),
        "inputs_manifest_sha256": sha256_file(inputs_manifest_path),
        "served_model": args.served_model, "url": args.url.rstrip("/"),
        "temperature": 0.0, "top_p": 1.0, "generation_seed": 0,
        "max_tokens": args.max_tokens, "limit": args.limit,
        "source_hashes": source_hashes(), "context_failure_policy": "abort; never shrink lists",
        "vllm_version": os.environ.get("TIE_VLLM_VERSION", "UNRECORDED"),
    }
    if protocol["vllm_version"] == "UNRECORDED":
        raise ValueError("Set TIE_VLLM_VERSION from the serving environment before evaluation")
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing evaluation protocol differs; use a fresh run directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Existing outputs have no protocol; refusing to mix runs")
        write_json(protocol_path, protocol)
    import requests
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ.get("LLM_API_KEY", "EMPTY")
    response = session.get(protocol["url"] + "/models", timeout=30)
    response.raise_for_status()
    if args.served_model not in {r["id"] for r in response.json()["data"]}:
        raise ValueError("Requested model is not advertised by this server")
    summaries = {}
    for retriever in ("bm25", "grit", "qwen3"):
        input_path = args.inputs / f"{retriever}.jsonl"
        if sha256_file(input_path) != inputs_manifest["files"][retriever]["input_sha256"]:
            raise ValueError("Frozen evaluation input changed")
        rows = read_jsonl(input_path)
        if args.limit:
            rows = rows[:args.limit]
        result_path = output / f"{retriever}.jsonl"
        done = read_jsonl(result_path) if result_path.exists() else []
        if len(done) > len(rows) or any(
            d["qid"] != r["qid"] or d["pre_reranked"] != r["pre_reranked"] or d["corpusids"] != r["corpusids"]
            for d, r in zip(done, rows)
        ):
            raise ValueError("Existing results are not an aligned prefix of frozen inputs")
        with result_path.open("a", encoding="utf-8") as handle:
            for row in rows[len(done):]:
                started = time.perf_counter()
                # Any HTTP/context error stops this run. Resume reruns that query.
                result = complete_query(session, protocol["url"], args.served_model, row, args.max_tokens)
                result["latency_seconds"] = time.perf_counter() - started
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                done.append(result)
                if len(done) % 25 == 0:
                    print(f"{retriever}: {len(done)}/{len(rows)}", flush=True)
        summary = {m: sum(r["metrics"][m] for r in done) / len(done) for m in done[0]["metrics"]}
        summary["queries"] = len(done)
        for metric in ("exact_permutation", "format_only", "fallback"):
            summary[metric + "_rate"] = sum(r["parser"][metric] for r in done) / len(done)
        for metric in ("missing_ids", "duplicate_ids", "invalid_ids"):
            summary[metric + "_rate"] = sum(r["parser"][metric] > 0 for r in done) / len(done)
            summary["mean_" + metric] = sum(r["parser"][metric] for r in done) / len(done)
        summary["length_finish_rate"] = sum(r["finish_reason"] == "length" for r in done) / len(done)
        summaries[retriever] = summary
    write_json(output / "summary.json", summaries)
    write_json(output / "complete.json", {"protocol_sha256": sha256_file(protocol_path), "results": {
        r: sha256_file(output / f"{r}.jsonl") for r in summaries
    }})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
