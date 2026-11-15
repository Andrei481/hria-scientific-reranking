"""Validate and freeze the shared adaptation supervision. No resampling."""
import argparse
from pathlib import Path
import shutil

from experiment import FROZEN_SHA256, audit_rows, read_jsonl, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    digest = sha256_file(args.source)
    if digest != FROZEN_SHA256:
        raise ValueError("Source is not the audited compact-baseline 4,494-list artifact; do not resample it")
    rows = read_jsonl(args.source)
    audit = audit_rows(rows)
    if len(rows) != 4494 or audit["list_size_histogram"] != {"50": 4312, "51": 182}:
        raise ValueError("Unexpected frozen dataset shape")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "lists.jsonl"
    if target.exists():
        if sha256_file(target) != digest:
            raise FileExistsError(f"Refusing to overwrite different data: {target}")
    elif args.source.resolve() != target.resolve():
        shutil.copyfile(args.source, target)
    if sha256_file(target) != digest:
        raise ValueError("Copy verification failed")
    manifest = {
        "data_sha256": digest, "source": str(args.source.resolve()), "audit": audit,
        "cohort": "new matched tie-sensitivity cohort; do not assume historical LLM list identity",
        "sampling": "none; actual 50/51-document lists and all relevance grades preserved",
    }
    write_json(args.output / "manifest.json", manifest)
    print(f"Verified {len(rows)} lists: {target}\nSHA256: {digest}")
    print(audit)


if __name__ == "__main__":
    main()
