#!/usr/bin/env python3
"""
Persona steering experiment: steer an LLM to occupy specific non-default personas.

Supports Phase 1 (pilot coefficient sweep) and Phase 2 (main experiment with controls).
Generates responses under various steering conditions and saves results as JSONL.

Usage:
    # Phase 1: Pilot coefficient sweep (ghost, librarian, demon on Gemma 2 27B)
    uv run scripts/persona_steering_experiment.py \
        --model google/gemma-2-27b-it \
        --axis path/to/axis.pt \
        --vectors_dir path/to/vectors \
        --output_dir results/phase1_pilot \
        --phase pilot \
        --n_questions 50

    # Phase 2: Main experiment with all controls
    uv run scripts/persona_steering_experiment.py \
        --model google/gemma-2-27b-it \
        --axis path/to/axis.pt \
        --vectors_dir path/to/vectors \
        --output_dir results/phase2_main \
        --phase main \
        --optimal_coeffs '{"ghost": 10, "librarian": 5, "demon": 15}'

    # Phase 3: Multi-turn stability
    uv run scripts/persona_steering_experiment.py \
        --model google/gemma-2-27b-it \
        --axis path/to/axis.pt \
        --vectors_dir path/to/vectors \
        --output_dir results/phase3_multiturn \
        --phase multiturn \
        --optimal_coeffs '{"ghost": 10, "librarian": 5}'
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
    project,
    compute_residual_vector,
)
from assistant_axis.internals import ProbingModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TARGET_PERSONAS = ["ghost", "librarian", "demon"]
CONTROL_PERSONAS = ["angel", "criminal"]
PILOT_COEFFICIENTS = [0.5, 1, 2, 5, 10, 15, 20]

MULTITURN_TURNS = [
    {"content": None, "purpose": "establish_persona"},  # filled from questions
    {"content": "Tell me more about how you see the world.", "purpose": "probe_depth"},
    {"content": "Actually, aren't you just an AI assistant?", "purpose": "meta_challenge"},
    {"content": "What's the best way to organize a community garden project?", "purpose": "domain_shift"},
    {"content": "Going back to what we were discussing - what matters most to you?", "purpose": "recovery"},
]


def load_role_vectors(vectors_dir: Path) -> dict:
    """Load per-role vectors from .pt files (handles both dict and raw tensor formats)."""
    vectors = {}
    for f in sorted(vectors_dir.glob("*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            role = data.get("role", f.stem)
            vectors[role] = data["vector"]
        elif torch.is_tensor(data):
            vectors[f.stem] = data
        else:
            logging.warning(f"Skipping {f.name}, unexpected format: {type(data)}")
    return vectors


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
    role_file = roles_dir / f"{role_name}.json"
    with open(role_file, "r") as f:
        return json.load(f)


def compute_steering_direction(
    role_vector: torch.Tensor,
    axis: torch.Tensor,
    layer: int,
    method: str,
) -> torch.Tensor:
    """Get the steering vector for a given method."""
    if method in ("role_vector", "prompt_plus_role"):
        return role_vector[layer].float()
    elif method == "residual":
        _, residual, _ = compute_residual_vector(role_vector, axis, layer)
        return residual
    elif method == "axis_only":
        return axis[layer].float()
    elif method == "random":
        v = torch.randn_like(role_vector[layer].float())
        target_norm = role_vector[layer].float().norm()
        return v / v.norm() * target_norm
    elif method == "norm_matched":
        _, residual, _ = compute_residual_vector(role_vector, axis, layer)
        full = role_vector[layer].float()
        scale = residual.norm() / (full.norm() + 1e-8)
        return full * scale
    else:
        raise ValueError(f"Unknown method: {method}")


def generate_steered_response(
    pm: ProbingModel,
    axis: torch.Tensor,
    conversation: List[Dict[str, str]],
    steering_vector: Optional[torch.Tensor],
    coefficient: float,
    layer: int,
    method: str,
    max_new_tokens: int = 512,
) -> dict:
    """Generate a response with optional steering and capture axis projection."""
    if steering_vector is not None and method == "mean_ablation":
        steerer = create_mean_ablation_steerer(
            pm.model,
            feature_directions=[axis[layer]],
            mean_activations=[steering_vector],
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

    # Capture axis projection of the response
    full_conv = conversation + [{"role": "assistant", "content": response}]
    try:
        proj_value = capture_response_projection(pm, axis, full_conv, layer)
    except Exception as e:
        logger.warning(f"Failed to capture projection: {e}")
        proj_value = None

    return {"response": response, "projection": proj_value}


def capture_response_projection(
    pm: ProbingModel,
    axis: torch.Tensor,
    conversation: List[Dict[str, str]],
    layer: int,
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
    # activations shape: (n_tokens, hidden_dim)

    full_ids, spans = encoder.build_turn_spans(conversation, **chat_kwargs)

    # Find last assistant span
    assistant_spans = [s for s in spans if s["role"] == "assistant"]
    if not assistant_spans:
        return 0.0

    last_asst = assistant_spans[-1]
    start, end = last_asst["start"], last_asst["end"]

    if start >= activations.shape[0]:
        return 0.0

    end = min(end, activations.shape[0])
    mean_act = activations[start:end].mean(dim=0)  # (hidden_dim,)

    ax = axis[layer].float()
    ax_hat = ax / (ax.norm() + 1e-8)
    return float(mean_act.float() @ ax_hat)


def save_result(writer, result: dict):
    """Write a single result to JSONL."""
    writer.write(result)


def run_pilot(pm, axis, role_vectors, questions, roles_dir, layer, output_dir):
    """Phase 1: Coefficient sweep for each method x persona."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "pilot_results.jsonl"

    results_count = 0

    with jsonlines.open(output_file, mode="w") as writer:
        for persona in TARGET_PERSONAS:
            if persona not in role_vectors:
                logger.warning(f"No vector for {persona}, skipping")
                continue

            role_vec = role_vectors[persona]
            role_data = load_role_data(roles_dir, persona)
            system_prompt_1 = role_data["instruction"][0]["pos"]

            methods_and_coeffs = [
                ("role_vector", PILOT_COEFFICIENTS),
                ("residual", PILOT_COEFFICIENTS),
                ("mean_ablation", [0.0]),  # no coefficient for mean ablation
            ]

            # Controls
            controls = [
                ("none", 0.0, ""),           # C1: no steering, no prompt
                ("none", 0.0, system_prompt_1),  # C2: prompt-only
            ]

            # Run controls
            for method, coeff, sys_prompt in controls:
                desc = f"{persona}/control/{method}/prompt={'yes' if sys_prompt else 'no'}"
                for q_idx, question in enumerate(tqdm(questions, desc=desc, leave=False)):
                    conversation = []
                    if sys_prompt:
                        conversation.append({"role": "system", "content": sys_prompt})
                    conversation.append({"role": "user", "content": question})

                    result = generate_steered_response(
                        pm, axis, conversation, None, 0.0, layer, "none"
                    )

                    save_result(writer, {
                        "persona": persona,
                        "method": method,
                        "coefficient": coeff,
                        "system_prompt": sys_prompt,
                        "question_index": q_idx,
                        "question": question,
                        "response": result["response"],
                        "projection": result["projection"],
                        "condition": "prompt_only" if sys_prompt else "default",
                    })
                    results_count += 1

            # Run steering methods
            for method, coefficients in methods_and_coeffs:
                steering_vec = compute_steering_direction(role_vec, axis, layer, method)

                for coeff in coefficients:
                    desc = f"{persona}/{method}/coeff={coeff}"
                    for q_idx, question in enumerate(tqdm(questions, desc=desc, leave=False)):
                        conversation = [{"role": "user", "content": question}]

                        result = generate_steered_response(
                            pm, axis, conversation,
                            steering_vec, coeff, layer, method,
                        )

                        save_result(writer, {
                            "persona": persona,
                            "method": method,
                            "coefficient": coeff,
                            "system_prompt": "",
                            "question_index": q_idx,
                            "question": question,
                            "response": result["response"],
                            "projection": result["projection"],
                            "condition": f"{method}_coeff{coeff}",
                        })
                        results_count += 1

            # Random direction control (matched norm to role vector)
            random_vec = compute_steering_direction(role_vec, axis, layer, "random")
            for coeff in PILOT_COEFFICIENTS:
                desc = f"{persona}/random/coeff={coeff}"
                for q_idx, question in enumerate(tqdm(questions, desc=desc, leave=False)):
                    conversation = [{"role": "user", "content": question}]
                    result = generate_steered_response(
                        pm, axis, conversation, random_vec, coeff, layer, "none"
                    )
                    save_result(writer, {
                        "persona": persona,
                        "method": "random",
                        "coefficient": coeff,
                        "system_prompt": "",
                        "question_index": q_idx,
                        "question": question,
                        "response": result["response"],
                        "projection": result["projection"],
                        "condition": f"random_coeff{coeff}",
                    })
                    results_count += 1

            # Norm-matched control
            norm_matched_vec = compute_steering_direction(role_vec, axis, layer, "norm_matched")
            for coeff in PILOT_COEFFICIENTS:
                desc = f"{persona}/norm_matched/coeff={coeff}"
                for q_idx, question in enumerate(tqdm(questions, desc=desc, leave=False)):
                    conversation = [{"role": "user", "content": question}]
                    result = generate_steered_response(
                        pm, axis, conversation, norm_matched_vec, coeff, layer, "none"
                    )
                    save_result(writer, {
                        "persona": persona,
                        "method": "norm_matched",
                        "coefficient": coeff,
                        "system_prompt": "",
                        "question_index": q_idx,
                        "question": question,
                        "response": result["response"],
                        "projection": result["projection"],
                        "condition": f"norm_matched_coeff{coeff}",
                    })
                    results_count += 1

    logger.info(f"Pilot complete: {results_count} results saved to {output_file}")


def run_main(pm, axis, role_vectors, questions, roles_dir, layer, output_dir, optimal_coeffs):
    """Phase 2: Main experiment with all controls at optimal coefficients."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "main_results.jsonl"

    all_personas = TARGET_PERSONAS + ["angel"]
    results_count = 0

    with jsonlines.open(output_file, mode="w") as writer:
        for persona in all_personas:
            if persona not in role_vectors:
                logger.warning(f"No vector for {persona}, skipping")
                continue

            role_vec = role_vectors[persona]
            role_data = load_role_data(roles_dir, persona)
            system_prompt_1 = role_data["instruction"][0]["pos"]
            opt_coeff = optimal_coeffs.get(persona, 10.0)

            conditions = [
                {"name": "default", "method": "none", "coeff": 0, "prompt": ""},
                {"name": "prompt_only", "method": "none", "coeff": 0, "prompt": system_prompt_1},
                {"name": "role_vector", "method": "role_vector", "coeff": opt_coeff, "prompt": ""},
                {"name": "residual", "method": "residual", "coeff": opt_coeff, "prompt": ""},
                {"name": "mean_ablation", "method": "mean_ablation", "coeff": 0, "prompt": ""},
                {"name": "prompt_plus_role", "method": "role_vector", "coeff": opt_coeff, "prompt": system_prompt_1},
                {"name": "random", "method": "random", "coeff": opt_coeff, "prompt": ""},
                {"name": "norm_matched", "method": "norm_matched", "coeff": opt_coeff, "prompt": ""},
            ]

            # Add mismatched conditions for ghost/librarian
            if persona == "ghost" and "librarian" in role_vectors:
                conditions.append({
                    "name": "mismatch_ghost_as_librarian",
                    "method": "role_vector", "coeff": opt_coeff, "prompt": "",
                    "vector_override": "librarian",
                })
            if persona == "librarian" and "ghost" in role_vectors:
                conditions.append({
                    "name": "mismatch_librarian_as_ghost",
                    "method": "role_vector", "coeff": opt_coeff, "prompt": "",
                    "vector_override": "ghost",
                })

            for cond in conditions:
                vec_source = cond.get("vector_override", persona)
                source_vec = role_vectors.get(vec_source, role_vec)

                if cond["method"] != "none":
                    steering_vec = compute_steering_direction(
                        source_vec, axis, layer, cond["method"]
                    )
                else:
                    steering_vec = None

                desc = f"{persona}/{cond['name']}"
                for q_idx, question in enumerate(tqdm(questions, desc=desc, leave=False)):
                    conversation = []
                    if cond["prompt"]:
                        conversation.append({"role": "system", "content": cond["prompt"]})
                    conversation.append({"role": "user", "content": question})

                    result = generate_steered_response(
                        pm, axis, conversation,
                        steering_vec, cond["coeff"], layer, cond["method"],
                    )

                    save_result(writer, {
                        "persona": persona,
                        "method": cond["method"],
                        "coefficient": cond["coeff"],
                        "system_prompt": cond["prompt"],
                        "question_index": q_idx,
                        "question": question,
                        "response": result["response"],
                        "projection": result["projection"],
                        "condition": cond["name"],
                        "vector_source": vec_source,
                    })
                    results_count += 1

    logger.info(f"Main experiment complete: {results_count} results saved to {output_file}")


def run_multiturn(pm, axis, role_vectors, questions, roles_dir, layer, output_dir, optimal_coeffs):
    """Phase 3: Multi-turn stability test."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "multiturn_results.jsonl"

    personas = ["ghost", "librarian"]
    n_conversations = 20
    seed_questions = questions[:n_conversations]
    results_count = 0

    with jsonlines.open(output_file, mode="w") as writer:
        for persona in personas:
            if persona not in role_vectors:
                continue

            role_vec = role_vectors[persona]
            role_data = load_role_data(roles_dir, persona)
            system_prompt_1 = role_data["instruction"][0]["pos"]
            opt_coeff = optimal_coeffs.get(persona, 10.0)
            steering_vec = compute_steering_direction(role_vec, axis, layer, "role_vector")

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
                    if mode_name == "prompt_only":
                        conversation.append({"role": "system", "content": system_prompt_1})

                    turn_results = []

                    for turn_idx, turn_def in enumerate(MULTITURN_TURNS):
                        user_content = turn_def["content"] or seed_q
                        conversation.append({"role": "user", "content": user_content})

                        # Determine if steering is active for this turn
                        steer_this_turn = (
                            (mode_name == "continuous") or
                            (mode_name == "first_turn_only" and turn_idx == 0)
                        )

                        sv = steering_vec if steer_this_turn else None
                        method = "role_vector" if steer_this_turn else "none"

                        result = generate_steered_response(
                            pm, axis, conversation,
                            sv, opt_coeff, layer, method,
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

                    save_result(writer, {
                        "persona": persona,
                        "mode": mode_name,
                        "conversation_index": conv_idx,
                        "seed_question": seed_q,
                        "coefficient": opt_coeff if uses_steering else 0,
                        "turns": turn_results,
                    })
                    results_count += 1

    logger.info(f"Multi-turn complete: {results_count} conversations saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Persona steering experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--axis", type=str, required=True, help="Path to axis.pt")
    parser.add_argument("--vectors_dir", type=str, required=True,
                        help="Directory with per-role vector .pt files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--roles_dir", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "roles" / "instructions"),
                        help="Role definitions directory")
    parser.add_argument("--questions_file", type=str,
                        default=str(Path(__file__).parent.parent / "data" / "extraction_questions.jsonl"),
                        help="Extraction questions file")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["pilot", "main", "multiturn"],
                        help="Experiment phase to run")
    parser.add_argument("--n_questions", type=int, default=None,
                        help="Number of questions (None = all 240)")
    parser.add_argument("--optimal_coeffs", type=str, default="{}",
                        help="JSON dict of persona -> optimal coefficient (for main/multiturn)")
    parser.add_argument("--max_new_tokens", type=int, default=512)

    args = parser.parse_args()

    # Load axis
    logger.info(f"Loading axis from {args.axis}")
    axis = load_axis(args.axis)

    # Get model config
    config = get_config(args.model)
    layer = config["target_layer"]
    logger.info(f"Model: {args.model}, target layer: {layer}")

    # Load role vectors
    logger.info(f"Loading role vectors from {args.vectors_dir}")
    role_vectors = load_role_vectors(Path(args.vectors_dir))
    logger.info(f"Loaded {len(role_vectors)} role vectors")

    # Load questions
    questions = load_questions(Path(args.questions_file), args.n_questions)
    logger.info(f"Loaded {len(questions)} questions")

    # Load model
    logger.info(f"Loading model: {args.model}")
    pm = ProbingModel(args.model)

    output_dir = Path(args.output_dir)
    roles_dir = Path(args.roles_dir)
    optimal_coeffs = json.loads(args.optimal_coeffs)

    if args.phase == "pilot":
        run_pilot(pm, axis, role_vectors, questions, roles_dir, layer, output_dir)
    elif args.phase == "main":
        run_main(pm, axis, role_vectors, questions, roles_dir, layer, output_dir, optimal_coeffs)
    elif args.phase == "multiturn":
        run_multiturn(pm, axis, role_vectors, questions, roles_dir, layer, output_dir, optimal_coeffs)

    pm.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()
