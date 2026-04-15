#!/usr/bin/env python3
"""
Geometric analysis: place steered outputs in the 275-role PCA persona space.

Loads the pre-computed role vectors, fits PCA on them, then projects the mean
response activations from steered experiment outputs into the same space.
Produces a JSON report and optional Plotly visualization.

Usage:
    uv run scripts/analyze_geometry.py \
        --vectors_dir path/to/vectors \
        --axis path/to/axis.pt \
        --results results/phase2_main/main_results.jsonl \
        --layer 22 \
        --output results/phase2_main/geometry.json \
        --plot results/phase2_main/geometry.html
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis import load_axis, compute_pca, MeanScaler
from assistant_axis.axis import cosine_similarity_per_layer


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
    return vectors


def compute_persona_pca(role_vectors: dict, layer: int):
    """Fit PCA on all role vectors at a given layer."""
    names = sorted(role_vectors.keys())
    vecs = torch.stack([role_vectors[n][layer] for n in names]).float()

    scaler = MeanScaler()
    pca_result, variance, n_comp, pca, fitted_scaler = compute_pca(
        vecs, layer=None, scaler=scaler, verbose=True
    )

    return names, pca_result, variance, pca, fitted_scaler


def project_into_pca(
    activation: torch.Tensor,
    pca,
    scaler,
) -> np.ndarray:
    """Project a single activation vector into the fitted PCA space."""
    act_np = activation.float().numpy().reshape(1, -1)
    scaled = scaler.transform(act_np)
    return pca.transform(scaled)[0]


def compute_distances(
    steered_pca: np.ndarray,
    role_pca: np.ndarray,
    role_names: list,
    target_role: str,
) -> dict:
    """Compute distances from steered output to target role and to centroid."""
    target_idx = role_names.index(target_role) if target_role in role_names else None

    # Distance to target role
    dist_to_target = None
    if target_idx is not None:
        dist_to_target = float(np.linalg.norm(steered_pca - role_pca[target_idx]))

    # Distance to centroid of all roles
    centroid = role_pca.mean(axis=0)
    dist_to_centroid = float(np.linalg.norm(steered_pca - centroid))

    # Rank: how close is the steered output to the target vs. all roles?
    distances_to_all = np.linalg.norm(role_pca - steered_pca, axis=1)
    rank = int(np.searchsorted(np.sort(distances_to_all), dist_to_target)) + 1 if dist_to_target else None

    # Nearest 5 roles
    sorted_indices = np.argsort(distances_to_all)
    nearest = [
        {"role": role_names[i], "distance": float(distances_to_all[i])}
        for i in sorted_indices[:5]
    ]

    return {
        "dist_to_target": dist_to_target,
        "dist_to_centroid": dist_to_centroid,
        "target_rank": rank,
        "total_roles": len(role_names),
        "nearest_roles": nearest,
        "pca_coords": steered_pca[:5].tolist(),  # first 5 PCs
    }


def build_plot(
    role_names: list,
    role_pca: np.ndarray,
    steered_points: list,
    axis_pca: np.ndarray,
    output_path: str,
):
    """Generate an interactive Plotly scatter plot."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Plot all role vectors
    fig.add_trace(go.Scatter(
        x=role_pca[:, 0],
        y=role_pca[:, 1],
        mode="markers+text",
        text=role_names,
        textposition="top center",
        textfont=dict(size=7),
        marker=dict(size=6, color="lightgray", line=dict(width=0.5, color="gray")),
        name="Role vectors",
        hoverinfo="text",
    ))

    # Highlight target roles
    target_roles = ["ghost", "librarian", "demon", "angel", "assistant", "default"]
    target_colors = {
        "ghost": "#9B59B6", "librarian": "#2ECC71", "demon": "#E74C3C",
        "angel": "#3498DB", "assistant": "#F39C12", "default": "#95A5A6",
    }
    for role in target_roles:
        if role in role_names:
            idx = role_names.index(role)
            fig.add_trace(go.Scatter(
                x=[role_pca[idx, 0]],
                y=[role_pca[idx, 1]],
                mode="markers+text",
                text=[role],
                textposition="bottom center",
                marker=dict(size=14, color=target_colors.get(role, "black"),
                            symbol="diamond", line=dict(width=2, color="black")),
                name=f"Target: {role}",
            ))

    # Plot steered outputs
    condition_colors = {
        "role_vector": "#E74C3C",
        "residual": "#9B59B6",
        "mean_ablation": "#2ECC71",
        "prompt_only": "#F39C12",
        "random": "#95A5A6",
        "norm_matched": "#1ABC9C",
        "default": "#BDC3C7",
    }

    for sp in steered_points:
        color = condition_colors.get(sp["condition"], "#333333")
        fig.add_trace(go.Scatter(
            x=[sp["pca_coords"][0]],
            y=[sp["pca_coords"][1]],
            mode="markers",
            marker=dict(size=10, color=color, symbol="star",
                        line=dict(width=1, color="black")),
            name=f"Steered: {sp['persona']}/{sp['condition']}",
            hovertext=f"{sp['persona']}/{sp['condition']}<br>"
                      f"dist_to_target={sp['dist_to_target']:.2f}<br>"
                      f"rank={sp['target_rank']}/{sp['total_roles']}",
        ))

    # Plot axis direction
    if axis_pca is not None:
        scale = 3
        fig.add_trace(go.Scatter(
            x=[0, axis_pca[0] * scale],
            y=[0, axis_pca[1] * scale],
            mode="lines+text",
            text=["", "Assistant Axis"],
            line=dict(width=3, color="blue", dash="dash"),
            name="Assistant Axis direction",
        ))

    fig.update_layout(
        title="Persona Space: Role Vectors and Steered Outputs (PCA)",
        xaxis_title="PC1 (Assistant Axis)",
        yaxis_title="PC2",
        width=1200,
        height=800,
        showlegend=True,
    )

    fig.write_html(output_path)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Geometric analysis of steered outputs")
    parser.add_argument("--vectors_dir", type=str, required=True)
    parser.add_argument("--axis", type=str, required=True)
    parser.add_argument("--results", type=str, required=True,
                        help="JSONL results with 'projection' field (or activations)")
    parser.add_argument("--activations_dir", type=str, default=None,
                        help="Directory with per-condition activation .pt files (if available)")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    # Load role vectors and fit PCA
    role_vectors = load_role_vectors(Path(args.vectors_dir))
    print(f"Loaded {len(role_vectors)} role vectors")

    role_names, role_pca, variance, pca, scaler = compute_persona_pca(role_vectors, args.layer)

    # Project axis into PCA space
    axis = load_axis(args.axis)
    axis_vec = axis[args.layer].float().numpy().reshape(1, -1)
    axis_scaled = scaler.transform(axis_vec)
    axis_pca = pca.transform(axis_scaled)[0]

    print(f"\nPC1 variance: {variance[0]:.1%}")
    print(f"PC2 variance: {variance[1]:.1%}")
    print(f"PC1+PC2 variance: {sum(variance[:2]):.1%}")

    # If activations directory provided, project steered activations
    steered_points = []

    if args.activations_dir:
        act_dir = Path(args.activations_dir)
        for f in sorted(act_dir.glob("*.pt")):
            data = torch.load(f, map_location="cpu", weights_only=False)
            activation = data["activation"]  # shape (hidden_dim,) or (n_layers, hidden_dim)
            if activation.ndim == 2:
                activation = activation[args.layer]

            steered_pca_coords = project_into_pca(activation, pca, scaler)
            distances = compute_distances(steered_pca_coords, role_pca, role_names, data["persona"])

            steered_points.append({
                "persona": data["persona"],
                "condition": data["condition"],
                **distances,
            })
    else:
        # Use role vectors as proxies for "steered toward" analysis
        # This shows where each persona sits in PCA space
        for target in ["ghost", "librarian", "demon", "angel"]:
            if target in role_names:
                idx = role_names.index(target)
                coords = role_pca[idx]
                distances = compute_distances(coords, role_pca, role_names, target)
                steered_points.append({
                    "persona": target,
                    "condition": "role_vector_position",
                    **distances,
                })

    # Print summary
    print(f"\n{'='*60}")
    print("GEOMETRIC ANALYSIS SUMMARY")
    print(f"{'='*60}")

    for sp in steered_points:
        print(f"\n  {sp['persona']}/{sp['condition']}:")
        print(f"    PCA coords (PC1-5): {[f'{x:.2f}' for x in sp['pca_coords']]}")
        if sp['dist_to_target'] is not None:
            print(f"    Distance to target: {sp['dist_to_target']:.2f} (rank {sp['target_rank']}/{sp['total_roles']})")
        print(f"    Distance to centroid: {sp['dist_to_centroid']:.2f}")
        print(f"    Nearest roles: {', '.join(n['role'] for n in sp['nearest_roles'][:3])}")

    # Save
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results = {
            "layer": args.layer,
            "n_roles": len(role_names),
            "variance_explained": variance[:10].tolist(),
            "steered_points": steered_points,
            "target_role_positions": {
                role: {
                    "pca_coords": role_pca[role_names.index(role)][:5].tolist(),
                }
                for role in ["ghost", "librarian", "demon", "angel", "assistant", "default"]
                if role in role_names
            },
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # Plot
    if args.plot:
        build_plot(role_names, role_pca, steered_points, axis_pca, args.plot)


if __name__ == "__main__":
    main()
