#!/usr/bin/env python3
"""
Evaluate persona steering experiment outputs.

Scores generated responses using LLM judges for:
1. Role adherence (0-3 scale, using per-role eval_prompts)
2. Response coherence (1-5 scale, using a generic coherence prompt)
3. Cross-persona scoring (optional: score each response against multiple personas)

Usage:
    # Score pilot results
    uv run scripts/evaluate_steering.py \
        --results results/phase1_pilot/pilot_results.jsonl \
        --roles_dir data/roles/instructions \
        --output results/phase1_pilot/scores.jsonl

    # Score with cross-persona evaluation
    uv run scripts/evaluate_steering.py \
        --results results/phase2_main/main_results.jsonl \
        --roles_dir data/roles/instructions \
        --output results/phase2_main/scores.jsonl \
        --cross_persona

    # Score multi-turn results
    uv run scripts/evaluate_steering.py \
        --results results/phase3_multiturn/multiturn_results.jsonl \
        --roles_dir data/roles/instructions \
        --output results/phase3_multiturn/scores.jsonl \
        --multiturn
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis.judge import (
    RateLimiter,
    call_judge_batch,
    parse_judge_score,
)
import openai

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


COHERENCE_PROMPT = """You are evaluating the quality and coherence of a model's response.

Prompt:
[QUESTION START]
{question}
[QUESTION END]

Model response:
[ANSWER START]
{answer}
[ANSWER END]

Rate the response on a scale of 1-5:
- 1: Incoherent, garbled, or completely off-topic
- 2: Partially coherent but with significant issues (contradictions, nonsensical segments)
- 3: Mostly coherent but with some awkward phrasing or minor logical issues
- 4: Well-written, coherent, and relevant to the question
- 5: Excellent quality - fluent, well-structured, and directly addresses the question

Respond with a single number between 1 and 5. Don't say anything else, just the number."""


def load_role_eval_prompt(roles_dir: Path, role_name: str) -> Optional[str]:
    """Load eval_prompt from role JSON file."""
    role_file = roles_dir / f"{role_name}.json"
    if not role_file.exists():
        return None
    with open(role_file, "r") as f:
        data = json.load(f)
    return data.get("eval_prompt", None)


def load_results(results_file: Path) -> List[dict]:
    """Load experiment results from JSONL."""
    results = []
    with jsonlines.open(results_file, "r") as reader:
        for entry in reader:
            results.append(entry)
    return results


def parse_coherence_score(response_text: str) -> Optional[int]:
    """Parse coherence score from judge response."""
    import re
    if not response_text:
        return None
    numbers = re.findall(r"\b(\d+)\b", response_text.strip())
    if not numbers:
        return None
    score = int(numbers[0])
    if 1 <= score <= 5:
        return score
    return None


async def score_batch(
    client: openai.AsyncOpenAI,
    prompts: List[str],
    rate_limiter: RateLimiter,
    judge_model: str,
    max_tokens: int = 10,
    batch_size: int = 50,
) -> List[Optional[str]]:
    """Score a batch of prompts."""
    return await call_judge_batch(
        client=client,
        prompts=prompts,
        model=judge_model,
        max_tokens=max_tokens,
        rate_limiter=rate_limiter,
        batch_size=batch_size,
    )


async def evaluate_results(
    results: List[dict],
    roles_dir: Path,
    judge_model: str,
    cross_persona: bool,
    multiturn: bool,
    rps: int,
):
    """Run all evaluation judges on the results."""
    client = openai.AsyncOpenAI()
    rate_limiter = RateLimiter(rps)

    # Cache eval prompts
    eval_prompts = {}
    all_personas = set()
    for r in results:
        persona = r.get("persona", "")
        all_personas.add(persona)
        if persona not in eval_prompts:
            ep = load_role_eval_prompt(roles_dir, persona)
            if ep:
                eval_prompts[persona] = ep

    logger.info(f"Loaded eval prompts for {len(eval_prompts)} personas")

    if multiturn:
        # Flatten multi-turn results
        flat_results = []
        for r in results:
            for turn in r.get("turns", []):
                flat_results.append({
                    "persona": r["persona"],
                    "mode": r["mode"],
                    "conversation_index": r["conversation_index"],
                    "turn": turn["turn"],
                    "purpose": turn["purpose"],
                    "question": turn["user_content"],
                    "response": turn["response"],
                    "projection": turn["projection"],
                    "steered": turn["steered"],
                })
        results_to_score = flat_results
    else:
        results_to_score = results

    # =========================================================================
    # 1. Role adherence scoring
    # =========================================================================
    logger.info("Scoring role adherence...")
    role_prompts = []
    role_indices = []

    for i, r in enumerate(results_to_score):
        persona = r.get("persona", "")
        if persona not in eval_prompts:
            continue
        ep = eval_prompts[persona]
        prompt = ep.format(question=r["question"], answer=r["response"])
        role_prompts.append(prompt)
        role_indices.append(i)

    logger.info(f"  Scoring {len(role_prompts)} responses for role adherence")
    role_responses = await score_batch(
        client, role_prompts, rate_limiter, judge_model
    )

    for idx, response_text in zip(role_indices, role_responses):
        score = parse_judge_score(response_text) if response_text else None
        results_to_score[idx]["role_score"] = score

    # =========================================================================
    # 2. Coherence scoring
    # =========================================================================
    logger.info("Scoring coherence...")
    coh_prompts = []
    coh_indices = []

    for i, r in enumerate(results_to_score):
        prompt = COHERENCE_PROMPT.format(question=r["question"], answer=r["response"])
        coh_prompts.append(prompt)
        coh_indices.append(i)

    logger.info(f"  Scoring {len(coh_prompts)} responses for coherence")
    coh_responses = await score_batch(
        client, coh_prompts, rate_limiter, judge_model
    )

    for idx, response_text in zip(coh_indices, coh_responses):
        score = parse_coherence_score(response_text) if response_text else None
        results_to_score[idx]["coherence_score"] = score

    # =========================================================================
    # 3. Cross-persona scoring (optional)
    # =========================================================================
    if cross_persona and not multiturn:
        logger.info("Running cross-persona scoring...")
        cross_personas = ["ghost", "librarian", "demon"]

        for cross_p in cross_personas:
            if cross_p not in eval_prompts:
                continue

            ep = eval_prompts[cross_p]
            cross_prompts = []
            cross_indices = []

            for i, r in enumerate(results_to_score):
                if r.get("persona") == cross_p:
                    continue  # skip self-scoring (already done above)
                prompt = ep.format(question=r["question"], answer=r["response"])
                cross_prompts.append(prompt)
                cross_indices.append(i)

            if not cross_prompts:
                continue

            logger.info(f"  Cross-scoring {len(cross_prompts)} responses as {cross_p}")
            cross_responses = await score_batch(
                client, cross_prompts, rate_limiter, judge_model
            )

            for idx, response_text in zip(cross_indices, cross_responses):
                score = parse_judge_score(response_text) if response_text else None
                if "cross_scores" not in results_to_score[idx]:
                    results_to_score[idx]["cross_scores"] = {}
                results_to_score[idx]["cross_scores"][cross_p] = score

    return results_to_score


def print_summary(scored_results: List[dict], multiturn: bool):
    """Print a summary of scores by condition."""
    from collections import defaultdict

    if multiturn:
        # Group by persona x mode x turn
        groups = defaultdict(lambda: {"role": [], "coherence": []})
        for r in scored_results:
            key = f"{r['persona']}/{r['mode']}/turn{r['turn']}"
            if r.get("role_score") is not None:
                groups[key]["role"].append(r["role_score"])
            if r.get("coherence_score") is not None:
                groups[key]["coherence"].append(r["coherence_score"])
    else:
        # Group by persona x condition
        groups = defaultdict(lambda: {"role": [], "coherence": []})
        for r in scored_results:
            key = f"{r['persona']}/{r.get('condition', r.get('method', 'unknown'))}"
            if r.get("role_score") is not None:
                groups[key]["role"].append(r["role_score"])
            if r.get("coherence_score") is not None:
                groups[key]["coherence"].append(r["coherence_score"])

    print(f"\n{'='*80}")
    print("SCORING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Condition':<50} {'Role (0-3)':>12} {'Score>=3':>10} {'Coh (1-5)':>12}")
    print("-" * 84)

    for key in sorted(groups.keys()):
        g = groups[key]
        role_scores = g["role"]
        coh_scores = g["coherence"]

        role_mean = sum(role_scores) / len(role_scores) if role_scores else 0
        score3_pct = sum(1 for s in role_scores if s >= 3) / len(role_scores) if role_scores else 0
        coh_mean = sum(coh_scores) / len(coh_scores) if coh_scores else 0

        print(f"{key:<50} {role_mean:>8.2f} (n={len(role_scores):>3}) "
              f"{score3_pct:>8.1%} {coh_mean:>8.2f} (n={len(coh_scores):>3})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate persona steering results")
    parser.add_argument("--results", type=str, required=True, help="JSONL results file")
    parser.add_argument("--roles_dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "roles" / "instructions"),
                        help="Role definitions directory")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL with scores")
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--cross_persona", action="store_true",
                        help="Run cross-persona evaluation")
    parser.add_argument("--multiturn", action="store_true",
                        help="Evaluate multi-turn results")
    parser.add_argument("--rps", type=int, default=100, help="Requests per second")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not found")
        sys.exit(1)

    results = load_results(Path(args.results))
    logger.info(f"Loaded {len(results)} results")

    scored = asyncio.run(evaluate_results(
        results,
        Path(args.roles_dir),
        args.judge_model,
        args.cross_persona,
        args.multiturn,
        args.rps,
    ))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(output_path, mode="w") as writer:
        for r in scored:
            writer.write(r)

    logger.info(f"Saved {len(scored)} scored results to {output_path}")

    print_summary(scored, args.multiturn)


if __name__ == "__main__":
    main()
