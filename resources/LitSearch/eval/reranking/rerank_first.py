import os
import copy
import argparse
import logging
import datasets
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List
from utils import utils
from datetime import datetime
import string

# ---------------------------
# Logging configuration
# ---------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join("logs", timestamp)
LOG_FILENAME = "reranker_first_debug.log"
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
# Document ID utilities
# ---------------------------

def get_doc_ids(n):
    """Generate document identifiers for n documents (A-Z, then AA, AB, ...)"""
    ids = list(string.ascii_uppercase)
    if n > 26:
        for c1 in string.ascii_uppercase:
            for c2 in string.ascii_uppercase:
                ids.append(c1 + c2)
                if len(ids) >= n:
                    return ids[:n]
    return ids[:n]

# ---------------------------
# FIRST Reranker Class
# ---------------------------

class FIRSTReranker:
    def __init__(self, model_path: str):
        logger.info(f"Loading FIRST model from {model_path}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        # Pre-compute token IDs for document identifiers
        self.max_docs = 200  # Support up to 200 documents
        doc_ids = get_doc_ids(self.max_docs)
        self.doc_letter_to_token = {}
        for letter in doc_ids:
            tokens = self.tokenizer.encode(letter, add_special_tokens=False)
            self.doc_letter_to_token[letter] = tokens[0]
        
        logger.info(f"Model loaded. Pre-computed {len(self.doc_letter_to_token)} document token IDs")
    
    def create_prompt(self, query: str, documents: List[str], max_words: int = 300) -> str:
        """Create FIRST-style prompt"""
        num_docs = len(documents)
        doc_ids = get_doc_ids(num_docs)
        
        passages_str_list = []
        for i, doc in enumerate(documents):
            content = " ".join(doc.strip().split()[:max_words])
            passages_str_list.append(f"[{doc_ids[i]}] {content}")
        
        docs_text = "\n\n".join(passages_str_list)
        
        prompt = f"""Query: {query}

Candidate Documents:
{docs_text}

Which document is most relevant to the query? Output only the document identifier."""
        
        return prompt
    
    def rerank(self, query: str, documents: List[str], max_words: int = 300) -> List[int]:
        """
        Rerank documents using logit extraction.
        Returns indices sorted by relevance (most relevant first).
        """
        num_docs = len(documents)
        doc_ids = get_doc_ids(num_docs)
        
        prompt = self.create_prompt(query, documents, max_words)
        
        messages = [{"role": "user", "content": prompt}]
        
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=16384)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        logger.debug(f"Input length: {inputs['input_ids'].shape[1]} tokens")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last position logits
        
        # Extract logits for each document identifier
        doc_token_ids = [self.doc_letter_to_token[doc_ids[i]] for i in range(num_docs)]
        doc_logits = logits[doc_token_ids]
        
        # Sort by logits (descending) to get ranking
        sorted_indices = torch.argsort(doc_logits, descending=True).cpu().tolist()
        
        logger.debug(f"Reranked {num_docs} documents. Top 5 indices: {sorted_indices[:5]}")
        
        return sorted_indices


def permutation_pipeline(model: FIRSTReranker, item: dict, rank_start: int, rank_end: int, index_type: str) -> dict:
    """
    Rerank documents using FIRST logit extraction.
    """
    query = item["query"]
    docs = item["documents"][rank_start:rank_end]
    
    if index_type == "title_abstract":
        max_words = 300
    elif index_type == "full_paper":
        max_words = 10000
    else:
        raise ValueError(f"Invalid index type: {index_type}")
    
    # Extract document contents
    doc_contents = [doc.get("content", "").strip() for doc in docs]
    
    try:
        # Get reranked indices
        sorted_indices = model.rerank(query, doc_contents, max_words)
        
        # Apply permutation
        reranked_docs = [copy.deepcopy(docs[i]) for i in sorted_indices]
        
        # Update item with reranked documents
        for j, doc in enumerate(reranked_docs):
            item["documents"][rank_start + j] = doc
        
        logger.debug(f"Successfully reranked query: {query[:50]}...")
        
    except Exception as e:
        logger.exception(f"Error during reranking: {e}")
        # Return original order on error
    
    return item


# ---------------------------
# MAIN SCRIPT
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_results_file", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to FIRST reranker model")
    parser.add_argument("--max_k", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="results/reranking/")
    parser.add_argument("--dataset_path", default="princeton-nlp/LitSearch")
    args = parser.parse_args()

    logger.info("Starting FIRST reranker with args: %s", vars(args))

    # Load FIRST model
    model = FIRSTReranker(args.model_path)

    # Load corpus and retrieval results
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

    # Limit to max_k documents
    for result in retrieval_results:
        result["retrieved"] = result.get("retrieved", [])[:args.max_k]

    # Prepare reranking inputs
    reranking_inputs = []
    for query_info in retrieval_results:
        reranking_inputs.append({
            "query": query_info.get("query", ""),
            "documents": [
                {"content": corpusid_to_text.get(cid, ""), "corpusid": cid}
                for cid in query_info.get("retrieved", [])
            ]
        })

    # Initialize output file
    if not os.path.exists(output_file):
        utils.write_json(copy.deepcopy(retrieval_results), output_file)

    # Process each query
    for item_idx, item in enumerate(tqdm(reranking_inputs, desc="Reranking")):
        try:
            reranking_outputs = utils.read_json(output_file)

            # Skip if already processed
            if "pre_reranked" in reranking_outputs[item_idx]:
                logger.info(f"Skipping already processed query idx {item_idx}")
                continue

            # Rerank using FIRST
            reranked_item = permutation_pipeline(
                model, item, 0, len(item["documents"]), index_type
            )

            # Save results
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

    logger.info("FIRST reranking completed. Results saved to: %s", output_file)
    logger.info("Logs written to: %s", log_path)    