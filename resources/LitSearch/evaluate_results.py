import json
import math
import os
import argparse
from typing import List, Dict, Any

# Used for Litsearch

def recall_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    topk = retrieved[:k]
    if len(relevant) == 0:
        return 0.0
    num_rel = len(set(topk).intersection(set(relevant)))
    return float(num_rel) / float(len(relevant))

def ndcg_at_k(retrieved: List[int], relevant: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    dcg = 0.0
    for idx, docid in enumerate(retrieved[:k]):
        if docid in rel_set:
            dcg += 1.0 / math.log2(idx + 2.0)
    ideal_rel_count = min(len(relevant), k)
    if ideal_rel_count == 0:
        return 0.0
    idcg = sum([1.0 / math.log2(i + 2.0) for i in range(ideal_rel_count)])
    return dcg / idcg if idcg > 0 else 0.0

def load_jsonl_or_json(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r') as f:
        first = f.readline()
        if not first:
            return []
        first = first.strip()
        if first.startswith('['):
            f.seek(0)
            data = json.load(f)
        else:
            try:
                data.append(json.loads(first))
            except json.JSONDecodeError:
                raise ValueError(f"Unable to parse first line of {path} as JSON")
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
    return data

def categorize(q: Dict[str, Any]) -> str:
    qset = q.get("query_set", "").lower()
    spec = int(q.get("specificity", 0))
    
    if "inline" in qset:
        prefix = "inline"
    elif qset in ("manual_acl", "manual_iclr"):
        prefix = "author"
    else:
        prefix = "other" # Safety fallback
        
    return f"{prefix}_broad" if spec == 0 else f"{prefix}_specific"

def safe_list_of_ints(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [int(i) for i in x]
    return [int(x)]

def aggregate_metrics(per_query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query_metrics:
        return {"R@20": 0.0, "R@5": 0.0, "NDCG@20": 0.0, "NDCG@5": 0.0}
    n = len(per_query_metrics)
    sum_r20 = sum(p["r20"] for p in per_query_metrics) / n
    sum_r5  = sum(p["r5"] for p in per_query_metrics) / n
    sum_n20 = sum(p["n20"] for p in per_query_metrics) / n
    sum_n5  = sum(p["n5"] for p in per_query_metrics) / n
    return {"R@20": sum_r20, "R@5": sum_r5, "NDCG@20": sum_n20, "NDCG@5": sum_n5}

def pretty_print_table(summary: Dict[str, Dict[str, float]]):
    rows = []
    keys = ["R@20", "R@5", "NDCG@20", "NDCG@5"]
    col_width = 16
    
    # MODIFIED HEADER: Added 'Count' column
    header = f"{'Category':<22}" + f"{'Count':>10}" + "".join([f"{k:>{col_width}}" for k in keys]) 
    print(header)
    print("-" * (22 + 10 + col_width * len(keys)))
    
    for cat, metrics in summary.items():
        row = f"{cat:<22}"
        
        count = metrics.get('count', 0)
        row += f"{count:>10}" 
        
        for k in keys:
            val = metrics.get(k, 0.0)
            row += f"{val*100:>{col_width}.2f}"
        print(row)

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval/reranking results for LitSearch-style JSONL")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input .json or .jsonl file")
    parser.add_argument("--ks", type=str, default="5,20", help="Comma separated k values to compute (default 5,20)")
    parser.add_argument("--save_json", action="store_true", default=True, help="Save JSON summary next to input file (default True)")
    args = parser.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    if not ks:
        ks = [5,20]

    data = load_jsonl_or_json(args.input)
    if not data:
        print("No data loaded from input.")
        return

    buckets = {
        "inline_broad": [],
        "inline_specific": [],
        "author_broad": [],
        "author_specific": [],
    }
    per_query_records = [] 

    for entry in data:
        retrieved = safe_list_of_ints(entry.get("retrieved", []))
        relevant  = safe_list_of_ints(entry.get("corpusids", []))
        cat = categorize(entry)
        
        perq = {}
        kmax = max(ks)
        perq["r20"] = recall_at_k(retrieved, relevant, 20 if 20 in ks else kmax)
        perq["r5"]  = recall_at_k(retrieved, relevant, 5 if 5 in ks else min(5, kmax))
        perq["n20"] = ndcg_at_k(retrieved, relevant, 20 if 20 in ks else kmax)
        perq["n5"]  = ndcg_at_k(retrieved, relevant, 5 if 5 in ks else min(5, kmax))
        per_query_records.append(perq)
        
        # Safe append to buckets (handling 'other' if needed)
        if cat not in buckets:
            buckets[cat] = []
        buckets[cat].append(perq)

    summary = {}
    
    # 1. Update basic categories
    for c in ["inline_broad", "inline_specific", "author_broad", "author_specific"]:
        bucket_data = buckets.get(c, [])
        summary[c] = aggregate_metrics(bucket_data)
        summary[c]['count'] = len(bucket_data)

    # 2. Update broad/specific averages
    broad_list = buckets.get("inline_broad", []) + buckets.get("author_broad", [])
    spec_list  = buckets.get("inline_specific", []) + buckets.get("author_specific", [])
    
    summary["broad_avg"] = aggregate_metrics(broad_list)
    summary["broad_avg"]['count'] = len(broad_list)
    
    summary["specific_avg"] = aggregate_metrics(spec_list)
    summary["specific_avg"]['count'] = len(spec_list)
    
    # 3. Update overall average
    summary["overall_avg"] = aggregate_metrics(per_query_records)
    summary["overall_avg"]['count'] = len(per_query_records)

    print("\nEvaluation results (values are percentages):\n")
    pretty_print_table(summary)

    if args.save_json:
        in_dir = os.path.dirname(args.input)
        stem = os.path.basename(args.input)
        for ext in [".jsonl", ".json"]:
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
        out_name = f"{stem}.metrics.json"
        out_path = os.path.join(in_dir if in_dir else ".", out_name)
        
        json_summary = {}
        for cat, metrics in summary.items():
            json_summary[cat] = {
                "count": metrics.get("count", 0),
                "R@20": metrics["R@20"],
                "R@5": metrics["R@5"],
                "NDCG@20": metrics["NDCG@20"],
                "NDCG@5": metrics["NDCG@5"],
                "R@20_percent": round(metrics["R@20"] * 100, 4),
                "R@5_percent": round(metrics["R@5"] * 100, 4),
                "NDCG@20_percent": round(metrics["NDCG@20"] * 100, 4),
                "NDCG@5_percent": round(metrics["NDCG@5"] * 100, 4),
            }
        with open(out_path, "w") as fw:
            json.dump(json_summary, fw, indent=2)
        print(f"\nSaved JSON summary to: {out_path}")

if __name__ == "__main__":
    main()