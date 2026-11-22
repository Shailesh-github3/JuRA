"""
scripts/evaluate_retrieval.py

Lightweight retrieval baseline for JuRA.

Runs every gold query in eval/gold_queries.json against the live Qdrant
collection through the same retrieve() the production path uses, and reports
Recall@1 / Recall@3 / Recall@5 and MRR, split by category (semantic vs.
exact section-number lookups). Failed queries are printed with the expected
provision and what was actually retrieved, so the decision of whether to add
BM25 + RRF or a reranker can be made from evidence rather than from vibes.

No framework required: only the stdlib and the repo's own dependencies.

Usage:
    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --k 10
    python scripts/evaluate_retrieval.py --gold eval/gold_queries.json --category exact
    python scripts/evaluate_retrieval.py --json > baseline.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.models.chunk import DocumentType, LegalStatus
from app.retrieval.retriever import retrieve
from app.vectorstore import qdrant_service
from app.vectorstore import qdrant_service


def _load_gold(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if "queries" not in data or not isinstance(data["queries"], list):
        raise ValueError("gold file must contain a 'queries' list")
    return data


def _coerce_enum(cls, value):
    if value is None:
        return None
    return cls(value)


def _unit(chunk):
    """The provision identifier a gold query compares against."""
    md = chunk.metadata
    if md.article:
        return ("article", md.article)
    if md.section:
        return ("section", md.section)
    return (None, None)


def _is_hit(chunk, expected):
    md = chunk.metadata
    if md.document_id != expected["document_id"]:
        return False
    if "article" in expected:
        return md.article == expected["article"]
    if "section" in expected:
        return md.section == expected["section"]
    return False


def _run_one(entry, k):
    document_type = _coerce_enum(DocumentType, entry.get("document_type"))
    legal_status = _coerce_enum(LegalStatus, entry.get("legal_status"))

    results = retrieve(
        query=entry["query"],
        limit=k,
        document_type=document_type,
        legal_status=legal_status,
    )

    rank = None
    for i, chunk in enumerate(results, start=1):
        if _is_hit(chunk, entry["expected"]):
            rank = i
            break

    return rank, results


def _validate_gold(entries, k_unused=None):
    """Check that every expected provision actually exists in the collection.

    A gold entry that points at a provision the corpus does not contain is a
    data bug, not a retrieval miss, and must be reported separately so the
    metrics below are not polluted by typos in the gold file.
    """
    from app.vectorstore import qdrant_service

    client = qdrant_service._get_client()
    missing = []
    for entry in entries:
        expected = entry["expected"]
        field = "article" if "article" in expected else "section"
        value = expected[field]

        must = [
            qdrant_service.models.FieldCondition(
                key="document_id",
                match=qdrant_service.models.MatchValue(value=expected["document_id"]),
            ),
            qdrant_service.models.FieldCondition(
                key=field,
                match=qdrant_service.models.MatchValue(value=value),
            ),
        ]
        points, _ = client.scroll(
            collection_name=qdrant_service.settings.QDRANT_COLLECTION_NAME,
            limit=1,
            with_payload=False,
            scroll_filter=qdrant_service.models.Filter(must=must),
        )
        if not points:
            missing.append(entry)
    return missing


def _metrics(ranks, k):
    """Return (recall@1, recall@3, recall@5-or-k, mrr)."""
    n = len(ranks)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    r1 = sum(1 for r in ranks if r is not None and r <= 1) / n
    r3 = sum(1 for r in ranks if r is not None and r <= 3) / n
    rk = sum(1 for r in ranks if r is not None and r <= min(k, 5)) / n
    # MRR: misses count as 0.0.
    mrr = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    return r1, r3, rk, mrr


def _print_table(rows):
    print(f"{'category':<10} {'n':>4} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7}")
    print("-" * 46)
    for name, (r1, r3, r5, mrr, n) in rows.items():
        print(f"{name:<10} {n:>4} {r1:>7.3f} {r3:>7.3f} {r5:>7.3f} {mrr:>7.3f}")


def main():
    parser = argparse.ArgumentParser(description="JuRA dense-retrieval baseline.")
    parser.add_argument("--gold", default=None, help="path to gold_queries.json")
    parser.add_argument("--k", type=int, default=5, help="retrieval depth (default 5)")
    parser.add_argument("--category", choices=("semantic", "exact"), default=None)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    gold_path = Path(args.gold) if args.gold else (root / "eval" / "gold_queries.json")
    data = _load_gold(gold_path)
    entries = data["queries"]
    if args.category:
        entries = [e for e in entries if e["category"] == args.category]

    # --- validate gold data first ---
    missing = _validate_gold(entries)
    if missing:
        print("WARNING: these gold entries point at provisions not in the collection "
              "(fix the gold file, do not tune retrieval to pass them):", file=sys.stderr)
        for e in missing:
            exp = e["expected"]
            print(f"  - {e['id']}: {exp.get('document_id')} "
                  f"{list(exp.keys())[1]}={list(exp.values())[1]}", file=sys.stderr)
        print("", file=sys.stderr)

    # --- run ---
    per_category = defaultdict(list)
    failures = []
    for entry in entries:
        rank, results = _run_one(entry, args.k)
        per_category[entry["category"]].append(rank)
        per_category["overall"].append(rank)

        if rank is None or rank > min(args.k, 5):
            failures.append((entry, results))

    # --- report ---
    if args.json:
        report = {
            "k": args.k,
            "gold_file": str(gold_path),
            "categories": {},
            "failures": [
                {
                    "id": e["id"],
                    "query": e["query"],
                    "expected": e["expected"],
                    "top": [
                        {
                            "document_id": c.metadata.document_id,
                            "article": c.metadata.article,
                            "section": c.metadata.section,
                        }
                        for c in res[: min(args.k, 5)]
                    ],
                }
                for e, res in failures
            ],
        }
        for name, ranks in per_category.items():
            r1, r3, r5, mrr = _metrics(ranks, args.k)
            report["categories"][name] = {
                "n": len(ranks),
                "recall@1": r1,
                "recall@3": r3,
                "recall@5": r5,
                "mrr": mrr,
            }
        print(json.dumps(report, indent=2))
        return

    print(f"JuRA retrieval baseline  |  k={args.k}  |  {len(entries)} queries  |  "
          f"gold: {gold_path}")
    print(f"MIN_SIMILARITY_SCORE = {qdrant_service.settings.MIN_SIMILARITY_SCORE}")
    print()

    rows = {}
    for name in ("semantic", "exact", "overall"):
        if name not in per_category:
            continue
        ranks = per_category[name]
        r1, r3, r5, mrr = _metrics(ranks, args.k)
        rows[name] = (r1, r3, r5, mrr, len(ranks))
    _print_table(rows)

    print()
    print(f"failed queries (expected provision not in top-{min(args.k, 5)}): {len(failures)}")
    for entry, results in failures:
        exp = entry["expected"]
        expected_label = f"{exp['document_id']} " + (
            f"article {exp['article']}" if "article" in exp else f"section {exp['section']}"
        )
        got = ", ".join(
            f"{c.metadata.document_id} "
            + (f"art {c.metadata.article}" if c.metadata.article else f"sec {c.metadata.section}")
            for c in results[: min(args.k, 5)]
        ) or "(nothing retrieved)"
        print(f"  - [{entry['id']}] {entry['query']!r}")
        print(f"      expected: {expected_label}")
        print(f"      got:      {got}")


if __name__ == "__main__":
    main()
