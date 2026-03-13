"""
5_hierarchical_prompt.py

Merges the RAG retrieval results (from 4_build_index.py) with the SpLiCE
concept keywords (from 3_run_splice.py) to produce the CEMRAG hierarchical
prompt JSON used for training / inference.

Output format (one entry per image):
  {
    "id": "...",
    "image": "...",
    "conversations": [
      {"from": "human", "value": "<hierarchical CEMRAG prompt>"},
      {"from": "gpt",   "value": "<ground-truth report>"}
    ]
  }

Usage
-----
  python scripts/5_hierarchical_prompt.py \\
      --rag   data/mimic-cxr_train_rag.json \\
      --splice data/mimic-cxr_train_spliceTerms.json \\
      --output data/mimic-cxr_train_cemrag.json
"""

import argparse
import json
import re
import sys


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

TASK_HEADER = (
    "<image>\n"
    "### TASK\n"
    "Write the report of the radiology image taking information from similar "
    "FINDINGS. Consider as more relevant sentences that contain any of the "
    "KEYWORDS in the FINDINGS.\n"
)

FOOTER = (
    "### END FINDINGS\n"
    "Write a paragraph with only the report relying in detail on the FINDINGS."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def extract_splice_keywords(human_value: str) -> list[str]:
    """Parse keywords embedded by 3_run_splice.py in the human message."""
    match = re.search(
        r"given these possible keywords:\s*(.*)", human_value, re.IGNORECASE
    )
    if not match:
        return []
    return [kw.strip() for kw in match.group(1).strip().split(",") if kw.strip()]


def extract_retrieved_reports(human_value: str) -> list[str]:
    """Parse numbered retrieved reports embedded by 4_build_index.py.

    Expected format in the RAG human message:
        "... 1) <report text>, 2) <report text>, ..."
    """
    matches = re.findall(r"\d+\)\s*(.*?)(?=,\s*\d+\)|$)", human_value, re.DOTALL)
    return [c.strip().rstrip(",.") for c in matches if c.strip()]


def build_cemrag_prompt(keywords: list[str], retrieved_reports: list[str]) -> str:
    prompt = TASK_HEADER + "\n"
    prompt += "### KEYWORDS: " + ", ".join(keywords) + "\n"
    prompt += "### FINDINGS\n"
    for i, report in enumerate(retrieved_reports, start=1):
        prompt += f"{i}. {report}\n"
    prompt += FOOTER
    return prompt.strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build CEMRAG hierarchical prompt JSON from RAG + SpLiCE outputs."
    )
    parser.add_argument(
        "--rag",
        required=True,
        metavar="PATH",
        help="RAG JSON produced by 4_build_index.py",
    )
    parser.add_argument(
        "--splice",
        required=True,
        metavar="PATH",
        help="SpLiCE JSON produced by 3_run_splice.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Output CEMRAG JSON path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading RAG file:    {args.rag}")
    data_rag = load_json(args.rag)

    print(f"Loading SpLiCE file: {args.splice}")
    data_splice = load_json(args.splice)

    # Build keyword lookup: id → [kw, ...]
    keywords_dict: dict[str, list[str]] = {}
    for item in data_splice:
        human_value = next(
            (msg["value"] for msg in item["conversations"] if msg["from"] == "human"),
            "",
        )
        keywords_dict[item["id"]] = extract_splice_keywords(human_value)

    missing_ids = 0
    for item in data_rag:
        item_id = item["id"]
        keywords = keywords_dict.get(item_id, [])
        if not keywords:
            missing_ids += 1

        for msg in item["conversations"]:
            if msg["from"] == "human":
                retrieved = extract_retrieved_reports(msg["value"])
                msg["value"] = build_cemrag_prompt(keywords, retrieved)
                break

    if missing_ids:
        print(
            f"Warning: {missing_ids}/{len(data_rag)} entries had no matching SpLiCE "
            "keywords (prompt built with empty KEYWORDS section).",
            file=sys.stderr,
        )

    with open(args.output, "w") as f:
        json.dump(data_rag, f, indent=4)

    print(f"Saved {len(data_rag)} entries to: {args.output}")


if __name__ == "__main__":
    main()
