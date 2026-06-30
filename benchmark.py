"""
evaluation/run_benchmark.py
---------------------------
Standalone retrieval benchmark.  No LLM is called — this evaluates the
retriever only, which is the correct way to validate Phase 1.

Usage
-----
From the project root (same directory as config.py):

    python evaluation/run_benchmark.py \\
        --video  "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" \\
        --benchmark evaluation/benchmark.json \\
        --top-k 5 10 \\
        --out evaluation/results.csv

Requirements
------------
    pip install rank-bm25 langchain langchain-community faiss-cpu \\
                sentence-transformers python-dotenv

The script imports your existing modules directly, so it must be run from
the project root.

Output
------
- Per-question results printed to stdout.
- Aggregate metrics (Recall@K, Precision@K, MRR, NDCG@K) printed and saved
  as a CSV file at --out path for tracking improvements over time.
"""

import sys
import csv
import json
import argparse
import importlib
from pathlib import Path
from typing import List, Dict, Any


# ---------------------------------------------------------------------------
# Make sure the project root is on the Python path so imports work.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from metrics import compute_aggregate_metrics, print_metrics_report  # noqa: E402


# ---------------------------------------------------------------------------
# Lazy imports — these pull from your project
# ---------------------------------------------------------------------------
def _load_project_modules():
    config_mod = importlib.import_module("config")
    settings = config_mod.load_settings()

    loader_mod = importlib.import_module("transcript_loader")
    vs_mod = importlib.import_module("vector_store")
    hr_mod = importlib.import_module("hybrid_retriever")

    return settings, loader_mod, vs_mod, hr_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _expand_with_tolerance(indices: List[int], tolerance: int) -> List[int]:
    """Expand a list of expected chunk indices to include ±tolerance neighbours.

    Used so that retrieving an adjacent chunk (e.g. 117 when 118 is expected)
    counts as a hit, since transcript chunks overlap by ~100 chars and the
    same explanation often spans neighbouring chunks.
    """
    if tolerance <= 0:
        return indices
    expanded = set()
    for idx in indices:
        for offset in range(-tolerance, tolerance + 1):
            expanded.add(idx + offset)
    expanded = {i for i in expanded if i > 0}
    return sorted(expanded)


def _load_benchmark(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Filter out comment/instruction objects.
    questions = [
        q for q in data
        if "question" in q and "id" in q
    ]
    return questions


def _build_retriever(video_urls: List[str], settings, loader_mod, vs_mod):
    """Load transcripts and build the vectorstore + HybridRetriever."""
    print(f"[benchmark] Loading transcripts for {len(video_urls)} video(s)...")
    videos, all_chunks, warnings = loader_mod.load_transcripts(video_urls, settings)

    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")

    print(f"[benchmark] {len(all_chunks)} chunks loaded across {len(videos)} video(s).")

    # Build embeddings (re-use the same logic as your main app).
    embeddings_mod = importlib.import_module("llm")
    embeddings = embeddings_mod.get_embeddings(settings)

    vectorstore, _retriever = vs_mod.build_vector_store(all_chunks, embeddings, settings)
    return vectorstore, videos, all_chunks


def _run_query(hybrid, question: Dict, top_k: int, tolerance: int = 0) -> Dict:
    """Retrieve for one benchmark question and return a result dict."""
    q_text = question["question"]
    expected = question.get("expected_chunk_indices", [])
    expected_for_scoring = _expand_with_tolerance(expected, tolerance)
    is_negative = question.get("is_negative", False)

    # Retrieve once and slice immediately so all downstream lists are consistent.
    scored = hybrid.retrieve_with_scores(q_text, top_k=max(top_k, 10))
    scored = scored[:top_k]  # single authoritative slice

    retrieved_indices = [
        doc.metadata.get("chunk_index", -1) for doc, _ in scored
    ]
    retrieved_timestamps = [
        doc.metadata.get("timestamp_range", doc.metadata.get("timestamp", ""))
        for doc, _ in scored
    ]
    retrieved_scores = [scores for _, scores in scored]

    # Check timestamp match if expected_timestamp is provided.
    expected_ts = question.get("expected_timestamp", "")
    timestamp_hit = False
    if expected_ts:
        timestamp_hit = any(expected_ts in ts for ts in retrieved_timestamps)

    return {
        "id": question["id"],
        "question": q_text,
        "is_negative": is_negative,
        "expected_chunk_indices": expected,
        "retrieved_indices": retrieved_indices,
        "retrieved_timestamps": retrieved_timestamps,
        "retrieved_scores": retrieved_scores,
        "timestamp_hit": timestamp_hit,
        "relevant_indices": expected_for_scoring,
    }


def _print_question_result(result: Dict, k_values: List[int]) -> None:
    from metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k, hit_at_k

    ri = result["retrieved_indices"]
    rel = result["relevant_indices"]
    neg = result["is_negative"]

    print(f"\n  [{result['id']}] {result['question']}")
    if neg:
        print("    Type    : NEGATIVE (no expected chunks)")
        print(f"    Retrieved (top {k_values[-1]}): {ri[:k_values[-1]]}")
        for k in k_values:
            prec = precision_at_k(ri, [], k)
            h = hit_at_k(ri, [], k)
            status = "PASS" if prec == 1.0 and h == 0.0 else "FAIL"
            print(
                f"    Precision@{k}={prec:.3f}  Hit@{k}={h:.0f}  [{status}]"
            )
    else:
        print(f"    Expected: {rel}")
        print(f"    Retrieved (top {k_values[-1]}): {ri[:k_values[-1]]}")
        for k in k_values:
            print(
                f"    Recall@{k}={recall_at_k(ri, rel, k):.3f}  "
                f"Precision@{k}={precision_at_k(ri, rel, k):.3f}  "
                f"NDCG@{k}={ndcg_at_k(ri, rel, k):.3f}  "
                f"Hit@{k}={hit_at_k(ri, rel, k):.0f}"
            )
        print(f"    MRR     : {reciprocal_rank(ri, rel):.3f}")
    if result.get("expected_timestamp"):
        hit = "HIT" if result["timestamp_hit"] else "MISS"
        print(f"    Timestamp {result['expected_timestamp']}: {hit}")

    # Show top-3 retrieved with scores.
    print("    Top-3 retrieved:")
    for idx, (chunk_idx, ts, scores) in enumerate(
        zip(
            result["retrieved_indices"][:3],
            result["retrieved_timestamps"][:3],
            result["retrieved_scores"][:3],
        ),
        start=1,
    ):
        bm = scores.get("bm25_score")
        dm = scores.get("dense_score")
        rrf = scores.get("rrf_score", 0.0)
        bm_str = f"{bm:.3f}" if bm is not None else "N/A"
        dm_str = f"{dm:.4f}" if dm is not None else "N/A"
        print(f"      {idx}. Chunk {chunk_idx}  {ts}  bm25={bm_str}  dense={dm_str}  rrf={rrf:.6f}")


def _save_csv(results: List[Dict], aggregate: Dict, out_path: str, k_values: List[int]) -> None:
    from metrics import recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k, hit_at_k

    rows = []
    for r in results:
        ri = r["retrieved_indices"]
        rel = r["relevant_indices"]
        row = {
            "id": r["id"],
            "question": r["question"],
            "is_negative": r["is_negative"],
            "expected_chunks": json.dumps(rel),
            "retrieved_chunks": json.dumps(ri[:10]),
        }
        for k in k_values:
            row[f"recall@{k}"] = recall_at_k(ri, rel, k)
            row[f"precision@{k}"] = precision_at_k(ri, rel, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ri, rel, k)
            row[f"hit@{k}"] = hit_at_k(ri, rel, k)
        row["mrr"] = reciprocal_rank(ri, rel)
        row["timestamp_hit"] = r.get("timestamp_hit", "")
        rows.append(row)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Append aggregate row.
    with open(out, "a", newline="", encoding="utf-8") as f:
        f.write("\n# AGGREGATE\n")
        for key, val in aggregate.items():
            f.write(f"# {key},{val}\n")

    print(f"\n[benchmark] Results saved to: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="RAG retrieval benchmark runner")
    parser.add_argument(
        "--video", nargs="+", required=True,
        help="One or more YouTube video URLs to load."
    )
    parser.add_argument(
        "--benchmark", default="evaluation/benchmark.json",
        help="Path to benchmark.json (default: /benchmark.json)"
    )
    parser.add_argument(
        "--top-k", nargs="+", type=int, default=[5, 10],
        help="K values for Recall/Precision/NDCG (default: 5 10)"
    )
    parser.add_argument(
        "--out", default="evaluation/results.csv",
        help="CSV output path (default: evaluation/results.csv)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable HybridRetriever verbose logging per query"
    )
    parser.add_argument(
        "--tolerance", type=int, default=0,
        help="±N adjacent chunk tolerance when scoring (default: 0, exact match). "
             "Use 1 to count neighbouring chunks as hits since chunks overlap."
    )
    args = parser.parse_args()

    k_values: List[int] = sorted(set(args.top_k))
    max_k = max(k_values)

    # ------------------------------------------------------------------
    print("\n[benchmark] Loading project modules...")
    settings, loader_mod, vs_mod, hr_mod = _load_project_modules()

    # ------------------------------------------------------------------
    vectorstore, videos, all_chunks = _build_retriever(
        args.video, settings, loader_mod, vs_mod
    )

    # Print chunk index map so you can fill in benchmark.json.
    
    all_docs = list(vectorstore.docstore._dict.values())

    chunk_map_file = Path("chunk_map.txt")

    with open(chunk_map_file, "w", encoding="utf-8") as f:
        f.write("[benchmark] Complete Chunk Index Map\n\n")

        for doc in sorted(all_docs, key=lambda d: d.metadata.get("chunk_index", 0)):
            idx = doc.metadata.get("chunk_index", "?")
            ts = doc.metadata.get("timestamp_range", "N/A")
            preview = doc.page_content[:80].replace("\n", " ")
    
            line = f"[{idx:>4}] {ts} | {preview}\n"
            f.write(line)

    print(f"[benchmark] Chunk map written to: {chunk_map_file.resolve()}")

    # ------------------------------------------------------------------
    print("\n[benchmark] Building HybridRetriever...")
    hybrid = hr_mod.HybridRetriever(vectorstore, settings, verbose=args.verbose)
    hybrid.build_index(all_docs)

    # ------------------------------------------------------------------
    questions = _load_benchmark(args.benchmark)
    print(f"\n[benchmark] Running {len(questions)} benchmark questions...")
    if args.tolerance > 0:
        print(f"[benchmark] Scoring with ±{args.tolerance} chunk tolerance (adjacent-chunk hits count).")

    results: List[Dict] = []
    for q in questions:
        result = _run_query(hybrid, q, top_k=max_k, tolerance=args.tolerance)
        results.append(result)
        _print_question_result(result, k_values)

    # ------------------------------------------------------------------
    aggregate = compute_aggregate_metrics(results, k_values=k_values)
    print_metrics_report(aggregate, title=f"Retrieval Benchmark — {len(questions)} questions")

    _save_csv(results, aggregate, args.out, k_values)

    # ------------------------------------------------------------------
    # Phase 1 status report
    # ------------------------------------------------------------------
    _print_phase1_status(aggregate, k_values, settings)


def _print_phase1_status(aggregate: Dict, k_values: List[int], settings: Any) -> None:
    """Print a weighted PASS/WARNING/FAIL summary for Phase 1 acceptance.

    Thresholds come from settings (config.py) and can be overridden via .env
    without touching benchmark code.

    Weighted scoring:
        Hit@K        — 30%  (did retrieval reach the correct topic?)
        Recall@K     — 30%  (how many relevant chunks were found?)
        MRR          — 20%  (was the first relevant chunk ranked early?)
        Neg Prec@K   — 20%  (did negative queries stay clean?)

    Overall score:
        >= 90%  PASS
        80-90%  WARNING  (review failing metrics before moving to Phase 2)
        < 80%   FAIL
    """
    min_k = min(k_values)

    hit_score    = aggregate.get(f"hit@{min_k}", 0.0)
    recall_score = aggregate.get(f"recall@{min_k}", 0.0)
    mrr_score    = aggregate.get("mrr", 0.0)
    neg_prec     = aggregate.get(f"neg_precision@{min_k}", 1.0)

    hit_target    = settings.phase1_hit_target
    recall_target = settings.phase1_recall_target
    mrr_target    = settings.phase1_mrr_target
    neg_target    = settings.phase1_neg_precision_target

    def _score(actual: float, target: float) -> float:
        """Fractional score: 1.0 at or above target, scales linearly below."""
        return min(1.0, actual / target) if target > 0 else 1.0

    weighted = (
        0.30 * _score(hit_score,    hit_target)
      + 0.30 * _score(recall_score, recall_target)
      + 0.20 * _score(mrr_score,    mrr_target)
      + 0.20 * _score(neg_prec,     neg_target)
    )
    overall_pct = weighted * 100

    checks = {
        f"Hit@{min_k}      {hit_score:.3f} / {hit_target}   (30%)":       hit_score    >= hit_target,
        f"Recall@{min_k}   {recall_score:.3f} / {recall_target}   (30%)": recall_score >= recall_target,
        f"MRR         {mrr_score:.3f} / {mrr_target}   (20%)":            mrr_score    >= mrr_target,
        f"Neg Prec@{min_k} {neg_prec:.3f} / {neg_target}   (20%)":        neg_prec     >= neg_target,
    }

    if overall_pct >= 90:
        verdict = "✅ OVERALL PASS"
    elif overall_pct >= 80:
        verdict = "⚠️  OVERALL WARNING"
    else:
        verdict = "❌ OVERALL FAIL"

    SEP = "=" * 60
    print(f"\n{SEP}")
    print("  PHASE 1 STATUS")
    print(SEP)
    for label, passed in checks.items():
        flag = "✅" if passed else "❌"
        print(f"  {flag}  {label}")
    print(SEP)
    print(f"  Weighted Score : {overall_pct:.1f}%")
    print(f"  {verdict}")
    print(SEP)


if __name__ == "__main__":
    main()  