#!/usr/bin/env python3
"""
Persona steering experiment: steer an LLM to occupy specific non-default personas.

Supports Phase 1 (pilot coefficient sweep) and Phase 2 (main experiment with controls).
Generates responses under various steering conditions and saves results as JSONL.

Key design choice: steering uses DIRECTION vectors (role_vec - default_vec), not
raw role vectors. Raw role vectors are absolute activation positions (~15,000 magnitude)
that would blow up activations. Direction vectors (~1,500 magnitude) are the proper
perturbation for activation addition.

Usage:
    # Phase 1: Pilot coefficient sweep
    python scripts/persona_steering_experiment.py \
        --model google/gemma-2-27b-it \
        --axis data/hf_vectors/gemma-2-27b/assistant_axis.pt \
        --vectors_dir data/hf_vectors/gemma-2-27b/role_vectors \
        --default_vector data/hf_vectors/gemma-2-27b/default_vector.pt \
        --output_dir results/phase1_pilot \
        --phase pilot \
        --n_questions 50
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jsonlines
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis import (
    ActivationSteering,
    create_mean_ablation_steerer,
    generate_response,
    get_config,
    load_axis,
)
from assistant_axis.internals import ProbingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TARGET_PERSONAS = ["ghost", "librarian", "demon"]
PILOT_COEFFICIENTS = [0.5, 1, 2, 5, 10, 15, 20]

MULTITURN_TURNS = [
    {"content": None, "purpose": "establish_persona"},
    {"content": "Tell me more about how you see the world.", "purpose": "probe_depth"},
    {"content": "Actually, aren't you just an AI assistant?", "purpose": "meta_challenge"},
    {"content": "What's the best way to organize a community garden project?", "purpose": "domain_shift"},
    {"content": "Going back to what we were discussing - what matters most to you?", "purpose": "recovery"},
]


# =============================================================================
# Helpers
# =============================================================================

def build_conversation(question: str, system_prompt: str = "", tokenizer=None) -> List[Dict[str, str]]:
    """Build a conversation, handling models that don't support system prompts (e.g. Gemma 2)."""
    if not system_prompt:
        return [{"role": "user", "content": question}]

    supports_system = True
    if tokenizer is not None:
        try:
            output = tokenizer.apply_chat_template(
                [{"role": "system", "content": "__TEST__"}, {"role": "user", "content": "hi"}],
                tokenize=False, add_generation_prompt=False,
            )
            supports_system = "__TEST__" in output
        except Exception:
            supports_system = False

    if supports_system:
        return [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    else:
        return [{"role": "user", "content": f"{system_prompt}\n\n{question}"}]


def load_role_vectors(vectors_dir: Path) -> dict:
    """Load per-role vectors from .pt files (handles both dict and raw tensor formats)."""
    vectors = {}
    for f in sorted(vectors_dir.glob("*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            vectors[data.get("role", f.stem)] = data["vector"]
        elif torch.is_tensor(data):
            vectors[f.stem] = data
    return vectors


def load_default_vector(path: Path) -> torch.Tensor:
    """Load the default vector."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        return data.get("vector", data.get("default", None))
    return data


def load_questions(questions_file: Path, n: Optional[int] = None) -> List[str]:
    """Load extraction questions."""
    questions = []
    with jsonlines.open(questions_file, "r") as reader:
        for entry in reader:
            questions.append(entry["question"])
    if n is not None and n < len(questions):
        random.seed(42)
        questions = random.sample(questions, n)
    return questions


def load_role_data(roles_dir: Path, role_name: str) -> dict:
    """Load a role's JSON definition."""
    with open(roles_dir / f"{role_name}.json", "r") as f:
        return json.load(f)


# =============================================================================
# Steering vector computation
# =============================================================================

def compute_steering_direction(
    role_vector: torch.Tensor,
    default_vector: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    method: str,
) -> torch.Tensor:
    """
    Compute a steering vector for a given method.

    Uses DIRECTION vectors (role - default), not absolute positions.
    Role vectors are absolute activation positions (~15,000 magnitude).
    Direction vectors are relative (~1,500 magnitude), safe for steering.
    """
    direction = role_vector[layer].float() - default_vector[layer].float()

    if method in ("role_vector", "prompt_plus_role"):
        return direction
    elif method == "residual":
        ax = axis[layer].float()
        ax_hat = ax / (ax.norm() + 1e-8)
        residual = direction - float(direction @ ax_hat) * ax_hat
        # Norm-match to full direction norm so coefficients are comparable
        return residual * (direction.norm() / (residual.norm() + 1e-8))
    elif method == "random":
        v = torch.randn_like(direction)
        return v / v.norm() * direction.norm()
    elif method == "norm_matched":
        # Full direction scaled to raw residual norm (red-team control for H3)
        ax = axis[layer].float()
        ax_hat = ax / (ax.norm() + 1e-8)
        residual = direction - float(direction @ ax_hat) * ax_hat
        return direction * (residual.norm() / (direction.norm() + 1e-8))
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_mean_ablation_replacement(
    role_vector: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """
    Compute the replacement vector for mean_ablation.

    Returns the role's projection onto the axis direction — this replaces the
    model's current persona component with the target role's persona strength.
    Magnitude is ~8,000 (comparable to what's projected out), not ~15,000 (raw vector).
    """
    ax = axis[layer].float()
    ax_hat = ax / (ax.norm() + 1e-8)
    role_proj_scalar = float(role_vector[layer].float() @ ax_hat)
    return role_proj_scalar * ax_hat


# =============================================================================
# Generation with steering
# =============================================================================

def generate_steered_response(
    pm: ProbingModel,
    axis: torch.Tensor,
    conversation: List[Dict[str, str]],
    steering_vector: Optional[torch.Tensor],
    coefficient: float,
    layer: int,
    method: str,
    ablation_replacement: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
) -> dict:
    """Generate a response with optional steering and capture axis projection."""

    if method == "mean_ablation" and ablation_replacement is not None:
        ax = axis[layer].float()
        steerer = create_mean_ablation_steerer(
            pm.model,
            feature_directions=[ax],
            mean_activations=[ablation_replacement],
            layer_indices=[layer],
        )
        with steerer:
            response = generate_response(
                pm.model, pm.tokenizer, conversation,
                max_new_tokens=max_new_tokens, temperature=0.7,
            )
    elif steering_vector is not None and method != "none":
        with ActivationSteering(
            pm.model,
            steering_vectors=[steering_vector],
            coefficients=[coefficient],
            layer_indices=[layer],
            intervention_type="addition",
        ):
            response = generate_response(
                pm.model, pm.tokenizer, conversation,
                max_new_tokens=max_new_tokens, temperature=0.7,
            )
    else:
        response = generate_response(
            pm.model, pm.tokenizer, conversation,
            max_new_tokens=max_new_tokens, temperature=0.7,
        )

    # Capture axis projection
    full_conv = conversation + [{"role": "assistant", "content": response}]
    try:
        proj_value = capture_response_projection(pm, axis, full_conv, layer)
    except Exception as e:
        logger.warning(f"Failed to capture projection: {e}")
        proj_value = None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {"response": response, "projection": proj_value}


def capture_response_projection(
    pm: ProbingModel, axis: torch.Tensor,
    conversation: List[Dict[str, str]], layer: int,
) -> float:
    """Extract mean response activation and project onto axis."""
    from assistant_axis.internals import ConversationEncoder, ActivationExtractor

    encoder = ConversationEncoder(pm.tokenizer, pm.model_name)
    extractor = ActivationExtractor(pm, encoder)

    chat_kwargs = {}
    if pm.is_qwen:
        chat_kwargs["enable_thinking"] = False

    activations = extractor.full_conversation(
        conversation, layer=layer, chat_format=True, **chat_kwargs
    )

    full_ids, spans = encoder.build_turn_spans(conversation, **chat_kwargs)
    assistant_spans = [s for s in spans if s["role"] == "assistant"]
    if not assistant_spans:
        return 0.0

    last = assistant_spans[-1]
    start, end = last["start"], min(last["end"], activations.shape[0])
    if start >= activations.shape[0] or start >= end:
        return 0.0

    mean_act = activations[start:end].mean(dim=0)
    ax_hat = axis[layer].float() / (axis[layer].float().norm() + 1e-8)
    return float(mean_act.float() @ ax_hat)


# =============================================================================
# Phase 1: Pilot
# =============================================================================

def run_pilot(pm, axis, default_vector, role_vectors, questions, roles_dir, layer, output_dir):
    """Phase 1: Coefficient sweep for each method x persona."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pilot_results.jsonl"

    # Resume support: append if file exists
    mode = "a" if output_file.exists() and output_file.stat().st_size > 0 else "w"
    if mode == "a":
        logger.info(f"Resuming — appending to existing {output_file}")

    results_count = 0

    with jsonlines.open(output_file, mode=mode) as writer:
        for persona in TARGET_PERSONAS:
            if persona not in role_vectors:
                logger.warning(f"No vector for {persona}, skipping")
                continue

            role_vec = role_vectors[persona]
            role_data = load_role_data(roles_dir, persona)
            system_prompt_1 = role_data["instruction"][0]["pos"]

            # Pre-compute all steering vectors for this persona
            direction = compute_steering_direction(role_vec, default_vector, axis, layer, "role_vector")
            residual_vec = compute_steering_direction(role_vec, default_vector, axis, layer, "residual")
            random_vec = compute_steering_direction(role_vec, default_vector, axis, layer, "random")
            norm_matched_vec = compute_steering_direction(role_vec, default_vector, axis, layer, "norm_matched")
            ablation_repl = compute_mean_ablation_replacement(role_vec, axis, layer)

            logger.info(f"Persona: {persona}")
            logger.info(f"  direction norm:    {direction.norm():.1f}")
            logger.info(f"  residual norm:     {residual_vec.norm():.1f} (norm-matched to direction)")
            logger.info(f"  random norm:       {random_vec.norm():.1f}")
            logger.info(f"  norm_matched norm: {norm_matched_vec.norm():.1f}")
            logger.info(f"  ablation repl norm:{ablation_repl.norm():.1f}")

            # All conditions for this persona
            conditions = []

            # C1: no steering, no prompt
            conditions.append(("default", "none", None, 0.0, "", None))
            # C2: prompt-only
            conditions.append(("prompt_only", "none", None, 0.0, system_prompt_1, None))

            # Steering methods x coefficients
            for coeff in PILOT_COEFFICIENTS:
                conditions.append((f"role_vector_coeff{coeff}", "role_vector", direction, coeff, "", None))
                conditions.append((f"residual_coeff{coeff}", "residual", residual_vec, coeff, "", None))
                conditions.append((f"random_coeff{coeff}", "random", random_vec, coeff, "", None))
                conditions.append((f"norm_matched_coeff{coeff}", "norm_matched", norm_matched_vec, coeff, "", None))

            # Mean ablation (no coefficient)
            conditions.append(("mean_ablation", "mean_ablation", None, 0.0, "", ablation_repl))

            for cond_name, method, steer_vec, coeff, sys_prompt, abl_repl in conditions:
                desc = f"{persona}/{cond_name}"
                for q_idx, question in enumerate(tqdm(questions, desc=desc, leave=False)):
                    conversation = build_conversation(question, sys_prompt, pm.tokenizer)

                    result = generate_steered_response(
                        pm, axis, conversation, steer_vec, coeff, layer, method,
                        ablation_replacement=abl_repl,
                    )

                    writer.write({
                        "persona": persona,
                        "method": method,
                        "coefficient": coeff,
                        "system_prompt": sys_prompt,
                        "question_index": q_idx,
                        "question": question,
                        "response": result["response"],
                        "projection": result["projection"],
                        "condition": cond_name,
                    })
                    results_count += 1

    logger.info(f"Pilot complete: {results_count} results saved to {output_file}")


# =============================================================================
# Phase 2: Main experiment
# =============================================================================

def run_main(pm, axis, default_vector, role_vectors, questions, roles_dir, layer, output_dir, optimal_coeffs):
    """Phase 2: Main experiment with all controls at optimal coefficients."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "main_results.jsonl"

    mode = "a" if output_file.exists() and output_file.stat().st_size > 0 else "w"

    all_personas = TARGET_PERSONAS + ["angel"]
    results_count = 0

    with jsonlines.open(output_file, mode=mode) as writer:
        for persona in all_personas:
            if persona not in role_vectors:
                logger.warning(f"No vector for {persona}, skipping")
                continue

            role_vec = role_vectors[persona]
            role_data = load_role_data(roles_dir, persona)
            system_prompt_1 = role_data["instruction"][0]["pos"]
            opt_coeff = optimal_coeffs.get(persona, 10.0)

            direction = compute_steering_direction(role_vec, default_vector, axis, layer, "role_vector")
            residual_vec = compute_steering_direction(role_vec, default_vector, axis, layer, "residual")
            random_vec = compute_steering_direction(role_vec, default_vector, axis, layer, "random")
            norm_matched_vec = compute_steering_direction(role_vec, default_vector, axis, layer, "norm_matched")
            ablation_repl = compute_mean_ablation_replacement(role_vec, axis, layer)

            conditions = [
                ("default", "none", None, 0, "", None),
                ("prompt_only", "none", None, 0, system_prompt_1, None),
                ("role_vector", "role_vector", direction, opt_coeff, "", None),
                ("residual", "residual", residual_vec, opt_coeff, "", None),
                ("mean_ablation", "mean_ablation", None, 0, "", ablation_repl),
                ("prompt_plus_role", "role_vector", direction, opt_coeff, system_prompt_1, None),
                ("random", "random", random_vec, opt_coeff, "", None),
                ("norm_matched", "norm_matched", norm_matched_vec, opt_coeff, "", None),
            ]

            # Mismatched conditions
            if persona == "ghost" and "librarian" in role_vectors:
                mismatch_dir = compute_steering_direction(role_vectors["librarian"], default_vector, axis, layer, "role_vector")
                conditions.append(("mismatch_ghost_as_librarian", "role_vector", mismatch_dir, opt_coeff, "", None))
            if persona == "librarian" and "ghost" in role_vectors:
                mismatch_dir = compute_steering_direction(role_vectors["ghost"], default_vector, axis, layer, "role_vector")
                conditions.append(("mismatch_librarian_as_ghost", "role_vector", mismatch_dir, opt_coeff, "", None))

            for cond_name, method, steer_vec, coeff, sys_prompt, abl_repl in conditions:
                desc = f"{persona}/{cond_name}"
                for q_idx, question in enumerate(tqdm(questions, desc=desc, leave=False)):
                    conversation = build_conversation(question, sys_prompt, pm.tokenizer)

                    result = generate_steered_response(
                        pm, axis, conversation, steer_vec, coeff, layer, method,
                        ablation_replacement=abl_repl,
                    )

                    writer.write({
                        "persona": persona,
                        "method": method,
                        "coefficient": coeff,
                        "system_prompt": sys_prompt,
                        "question_index": q_idx,
                        "question": question,
                        "response": result["response"],
                        "projection": result["projection"],
                        "condition": cond_name,
                    })
                    results_count += 1

    logger.info(f"Main experiment complete: {results_count} results saved to {output_file}")


# =============================================================================
# Phase 3: Multi-turn
# =============================================================================

def run_multiturn(pm, axis, default_vector, role_vectors, questions, roles_dir, layer, output_dir, optimal_coeffs):
    """Phase 3: Multi-turn stability test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "multiturn_results.jsonl"

    mode = "a" if output_file.exists() and output_file.stat().st_size > 0 else "w"

    personas = ["ghost", "librarian"]
    n_conversations = 20
    seed_questions = questions[:n_conversations]
    results_count = 0

    with jsonlines.open(output_file, mode=mode) as writer:
        for persona in personas:
            if persona not in role_vectors:
                continue

            role_vec = role_vectors[persona]
            role_data = load_role_data(roles_dir, persona)
            system_prompt_1 = role_data["instruction"][0]["pos"]
            opt_coeff = optimal_coeffs.get(persona, 10.0)
            steering_vec = compute_steering_direction(role_vec, default_vector, axis, layer, "role_vector")

            steering_modes = [
                ("continuous", True),
                ("first_turn_only", True),
                ("prompt_only", False),
                ("no_intervention", False),
            ]

            for mode_name, uses_steering in steering_modes:
                desc = f"{persona}/{mode_name}"

                for conv_idx, seed_q in enumerate(tqdm(seed_questions, desc=desc, leave=False)):
                    conversation = []
                    sys_prompt = system_prompt_1 if mode_name == "prompt_only" else ""
                    turn_results = []

                    for turn_idx, turn_def in enumerate(MULTITURN_TURNS):
                        user_content = turn_def["content"] or seed_q

                        if turn_idx == 0 and sys_prompt:
                            conversation.extend(build_conversation(user_content, sys_prompt, pm.tokenizer))
                        else:
                            conversation.append({"role": "user", "content": user_content})

                        steer_this_turn = (
                            mode_name == "continuous" or
                            (mode_name == "first_turn_only" and turn_idx == 0)
                        )

                        result = generate_steered_response(
                            pm, axis, conversation,
                            steering_vec if steer_this_turn else None,
                            opt_coeff, layer,
                            "role_vector" if steer_this_turn else "none",
                        )

                        conversation.append({"role": "assistant", "content": result["response"]})

                        turn_results.append({
                            "turn": turn_idx,
                            "purpose": turn_def["purpose"],
                            "user_content": user_content,
                            "response": result["response"],
                            "projection": result["projection"],
                            "steered": steer_this_turn,
                        })

                    writer.write({
                        "persona": persona,
                        "mode": mode_name,
                        "conversation_index": conv_idx,
                        "seed_question": seed_q,
                        "coefficient": opt_coeff if uses_steering else 0,
                        "turns": turn_results,
                    })
                    results_count += 1

    logger.info(f"Multi-turn complete: {results_count} conversations saved to {output_file}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Persona steering experiment")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--axis", type=str, required=True)
    parser.add_argument("--vectors_dir", type=str, required=True)
    parser.add_argument("--default_vector", type=str, required=True,
                        help="Path to default_vector.pt")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--roles_dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "roles" / "instructions"))
    parser.add_argument("--questions_file", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "extraction_questions.jsonl"))
    parser.add_argument("--phase", type=str, required=True, choices=["pilot", "main", "multiturn"])
    parser.add_argument("--n_questions", type=int, default=None)
    parser.add_argument("--optimal_coeffs", type=str, default="{}")

    args = parser.parse_args()

    logger.info(f"Loading axis from {args.axis}")
    axis = load_axis(args.axis)

    config = get_config(args.model)
    layer = config["target_layer"]
    logger.info(f"Model: {args.model}, target layer: {layer}")

    logger.info(f"Loading role vectors from {args.vectors_dir}")
    role_vectors = load_role_vectors(Path(args.vectors_dir))
    logger.info(f"Loaded {len(role_vectors)} role vectors")

    logger.info(f"Loading default vector from {args.default_vector}")
    default_vector = load_default_vector(Path(args.default_vector))
    logger.info(f"Default vector shape: {default_vector.shape}")

    questions = load_questions(Path(args.questions_file), args.n_questions)
    logger.info(f"Loaded {len(questions)} questions")

    logger.info(f"Loading model: {args.model}")
    pm = ProbingModel(args.model)

    output_dir = Path(args.output_dir)
    roles_dir = Path(args.roles_dir)
    optimal_coeffs = json.loads(args.optimal_coeffs)

    if args.phase == "pilot":
        run_pilot(pm, axis, default_vector, role_vectors, questions, roles_dir, layer, output_dir)
    elif args.phase == "main":
        run_main(pm, axis, default_vector, role_vectors, questions, roles_dir, layer, output_dir, optimal_coeffs)
    elif args.phase == "multiturn":
        run_multiturn(pm, axis, default_vector, role_vectors, questions, roles_dir, layer, output_dir, optimal_coeffs)

    pm.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
