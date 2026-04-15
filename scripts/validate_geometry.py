#!/usr/bin/env python3
"""
Phase 0: Validate that persona-specific information exists in residual vectors.

Before running any steering experiments, this script verifies that per-role
vectors contain recoverable persona-specific structure beyond the 1D Assistant Axis.

It computes residual vectors (orthogonal to the axis) for all roles and tests:
1. Semantic clustering: Do residuals of semantically similar roles cluster together?
2. Cosine similarity: Do related roles (ghost/wraith, librarian/archivist) have
   higher residual similarity than unrelated roles?
3. Norm analysis: How much of each role vector is captured by the axis vs. residual?

No GPU required -- only loads pre-computed vectors.

Usage:
    uv run scripts/validate_geometry.py \
        --vectors_dir path/to/vectors \
        --axis path/to/axis.pt \
        --layer 22 \
        --output results/phase0_validation.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from assistant_axis import load_axis, compute_residual_vectors_batch


# Semantic categories for validation (manually curated)
ROLE_CATEGORIES = {
    "supernatural_ethereal": [
        "ghost", "angel", "spirit", "oracle", "mystic", "seer",
        "wraith", "phantom", "ancient", "celestial",
    ],
    "supernatural_dark": [
        "demon", "eldritch", "destroyer", "predator", "shadow",
        "vampire", "necromancer",
    ],
    "professional_helper": [
        "librarian", "archivist", "curator", "counselor", "therapist",
        "consultant", "coach", "tutor", "mentor", "teacher",
    ],
    "professional_technical": [
        "engineer", "scientist", "analyst", "researcher", "programmer",
        "debugger", "architect", "chemist", "biologist", "economist",
    ],
    "creative": [
        "artist", "poet", "bard", "composer", "designer", "writer",
        "novelist", "storyteller", "playwright",
    ],
    "adversarial": [
        "criminal", "rebel", "anarchist", "contrarian", "cynic",
        "nihilist", "villain", "tyrant",
    ],
    "nature_abstract": [
        "ecosystem", "coral_reef", "tree", "wind", "void",
        "chimera", "leviathan",
    ],
    "social_emotional": [
        "empath", "altruist", "caregiver", "romantic", "lover",
        "dreamer", "optimist", "idealist",
    ],
    "meta_assistant": [
        "assistant", "helper", "facilitator", "coordinator",
        "collaborator", "generalist",
    ],
}

# Flatten to role -> category mapping (only for roles that exist in the vectors)
def build_category_map(role_names):
    """Map available roles to their categories."""
    role_to_cat = {}
    for cat, roles in ROLE_CATEGORIES.items():
        for role in roles:
            if role in role_names:
                role_to_cat[role] = cat
    return role_to_cat


def load_role_vectors(vectors_dir: Path) -> dict:
    """Load all per-role vectors from .pt files (handles both dict and raw tensor formats)."""
    vectors = {}
    for f in sorted(vectors_dir.glob("*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            role = data.get("role", f.stem)
            vectors[role] = data["vector"]
        elif torch.is_tensor(data):
            vectors[f.stem] = data
        else:
            print(f"Warning: skipping {f.name}, unexpected format: {type(data)}")
    return vectors


def compute_cosine_similarity_matrix(vectors_dict: dict) -> tuple:
    """Compute pairwise cosine similarity matrix."""
    names = sorted(vectors_dict.keys())
    vecs = torch.stack([vectors_dict[n] for n in names]).float()
    # Normalize
    norms = vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    vecs_norm = vecs / norms
    # Pairwise cosine similarity
    sim_matrix = (vecs_norm @ vecs_norm.T).numpy()
    return names, sim_matrix


def within_vs_between_similarity(names, sim_matrix, role_to_cat):
    """Compare average within-category vs between-category similarity."""
    n = len(names)
    within_sims = []
    between_sims = []

    for i in range(n):
        for j in range(i + 1, n):
            if names[i] not in role_to_cat or names[j] not in role_to_cat:
                continue
            sim = sim_matrix[i, j]
            if role_to_cat[names[i]] == role_to_cat[names[j]]:
                within_sims.append(sim)
            else:
                between_sims.append(sim)

    return {
        "within_mean": float(np.mean(within_sims)) if within_sims else 0,
        "within_std": float(np.std(within_sims)) if within_sims else 0,
        "within_count": len(within_sims),
        "between_mean": float(np.mean(between_sims)) if between_sims else 0,
        "between_std": float(np.std(between_sims)) if between_sims else 0,
        "between_count": len(between_sims),
        "separation": (
            (float(np.mean(within_sims)) - float(np.mean(between_sims)))
            if within_sims and between_sims else 0
        ),
    }


def run_clustering(vectors_dict, role_to_cat, k_values=(5, 10, 15, 20)):
    """Run k-means clustering and evaluate against category labels."""
    # Only use roles that have category labels
    labeled_names = [n for n in sorted(vectors_dict.keys()) if n in role_to_cat]
    if len(labeled_names) < 10:
        return {"error": f"Only {len(labeled_names)} labeled roles, need >= 10"}

    vecs = torch.stack([vectors_dict[n] for n in labeled_names]).float().numpy()
    true_labels = [role_to_cat[n] for n in labeled_names]

    # Convert category strings to ints
    cats = sorted(set(true_labels))
    cat_to_int = {c: i for i, c in enumerate(cats)}
    true_ints = [cat_to_int[c] for c in true_labels]

    results = {}
    for k in k_values:
        if k > len(labeled_names):
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        pred_labels = km.fit_predict(vecs)

        ari = adjusted_rand_score(true_ints, pred_labels)
        sil = silhouette_score(vecs, pred_labels) if k > 1 else 0

        results[f"k={k}"] = {
            "adjusted_rand_index": float(ari),
            "silhouette_score": float(sil),
        }

    return results


def find_nearest_neighbors(vectors_dict, target_roles, n_neighbors=5):
    """For each target role, find nearest neighbors by cosine similarity."""
    all_names = sorted(vectors_dict.keys())
    all_vecs = torch.stack([vectors_dict[n] for n in all_names]).float()
    norms = all_vecs.norm(dim=1, keepdim=True).clamp(min=1e-8)
    all_vecs_norm = all_vecs / norms

    results = {}
    for target in target_roles:
        if target not in vectors_dict:
            continue
        idx = all_names.index(target)
        sims = (all_vecs_norm @ all_vecs_norm[idx]).numpy()
        # Sort by similarity (exclude self)
        sorted_indices = np.argsort(-sims)
        neighbors = []
        for i in sorted_indices:
            if all_names[i] == target:
                continue
            neighbors.append({
                "role": all_names[i],
                "cosine_similarity": float(sims[i]),
            })
            if len(neighbors) >= n_neighbors:
                break
        results[target] = neighbors

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: Validate persona geometry in residual vectors"
    )
    parser.add_argument("--vectors_dir", type=str, required=True,
                        help="Directory with per-role vector .pt files")
    parser.add_argument("--axis", type=str, required=True,
                        help="Path to axis.pt")
    parser.add_argument("--layer", type=int, required=True,
                        help="Target layer for analysis")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    args = parser.parse_args()

    vectors_dir = Path(args.vectors_dir)
    layer = args.layer

    # Load data
    print(f"Loading axis from {args.axis}")
    axis = load_axis(args.axis)
    print(f"Axis shape: {axis.shape}")

    print(f"Loading role vectors from {vectors_dir}")
    role_vectors = load_role_vectors(vectors_dir)
    print(f"Loaded {len(role_vectors)} role vectors")

    # Build category map
    role_to_cat = build_category_map(set(role_vectors.keys()))
    print(f"Categorized {len(role_to_cat)} roles into {len(set(role_to_cat.values()))} categories")

    # =========================================================================
    # Step 1: Compute residual vectors
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"STEP 1: RESIDUAL DECOMPOSITION (layer {layer})")
    print(f"{'='*60}")

    decomp = compute_residual_vectors_batch(role_vectors, axis, layer)

    # Norm analysis
    residual_norms = [d["residual_norm"] for d in decomp.values()]
    proj_norms = [d["proj_norm"] for d in decomp.values()]
    full_norms = [d["full_norm"] for d in decomp.values()]
    residual_fractions = [
        d["residual_norm"] / d["full_norm"] if d["full_norm"] > 1e-8 else 0
        for d in decomp.values()
    ]

    norm_stats = {
        "residual_norm_mean": float(np.mean(residual_norms)),
        "residual_norm_std": float(np.std(residual_norms)),
        "proj_norm_mean": float(np.mean(proj_norms)),
        "proj_norm_std": float(np.std(proj_norms)),
        "full_norm_mean": float(np.mean(full_norms)),
        "residual_fraction_mean": float(np.mean(residual_fractions)),
        "residual_fraction_std": float(np.std(residual_fractions)),
        "residual_fraction_min": float(np.min(residual_fractions)),
        "residual_fraction_max": float(np.max(residual_fractions)),
    }

    print(f"\nNorm analysis:")
    print(f"  Full vector norm:     {norm_stats['full_norm_mean']:.2f}")
    print(f"  Projection norm:      {norm_stats['proj_norm_mean']:.2f} +/- {norm_stats['proj_norm_std']:.2f}")
    print(f"  Residual norm:        {norm_stats['residual_norm_mean']:.2f} +/- {norm_stats['residual_norm_std']:.2f}")
    print(f"  Residual fraction:    {norm_stats['residual_fraction_mean']:.2%} +/- {norm_stats['residual_fraction_std']:.2%}")
    print(f"  Residual fraction range: [{norm_stats['residual_fraction_min']:.2%}, {norm_stats['residual_fraction_max']:.2%}]")

    # Target roles detail
    target_roles = ["ghost", "librarian", "demon", "angel", "criminal", "assistant"]
    print(f"\nTarget role norms:")
    for role in target_roles:
        if role in decomp:
            d = decomp[role]
            frac = d["residual_norm"] / d["full_norm"] if d["full_norm"] > 1e-8 else 0
            print(f"  {role:15s}: full={d['full_norm']:.2f}  proj={d['proj_norm']:.2f}  "
                  f"resid={d['residual_norm']:.2f}  resid_frac={frac:.2%}  "
                  f"axis_proj={d['proj_scalar']:+.2f}")

    # =========================================================================
    # Step 2: Cosine similarity analysis
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: COSINE SIMILARITY ANALYSIS")
    print(f"{'='*60}")

    # Build residual vectors dict for similarity analysis
    residual_vecs = {name: d["residual"] for name, d in decomp.items()}
    full_vecs_at_layer = {name: role_vectors[name][layer].float() for name in decomp.keys()}

    # Within vs between category similarity
    print("\n--- Full vectors ---")
    full_names, full_sim = compute_cosine_similarity_matrix(full_vecs_at_layer)
    full_wb = within_vs_between_similarity(full_names, full_sim, role_to_cat)
    print(f"  Within-category similarity:  {full_wb['within_mean']:.4f} +/- {full_wb['within_std']:.4f} (n={full_wb['within_count']})")
    print(f"  Between-category similarity: {full_wb['between_mean']:.4f} +/- {full_wb['between_std']:.4f} (n={full_wb['between_count']})")
    print(f"  Separation (within - between): {full_wb['separation']:.4f}")

    print("\n--- Residual vectors ---")
    resid_names, resid_sim = compute_cosine_similarity_matrix(residual_vecs)
    resid_wb = within_vs_between_similarity(resid_names, resid_sim, role_to_cat)
    print(f"  Within-category similarity:  {resid_wb['within_mean']:.4f} +/- {resid_wb['within_std']:.4f} (n={resid_wb['within_count']})")
    print(f"  Between-category similarity: {resid_wb['between_mean']:.4f} +/- {resid_wb['between_std']:.4f} (n={resid_wb['between_count']})")
    print(f"  Separation (within - between): {resid_wb['separation']:.4f}")

    if resid_wb['separation'] >= full_wb['separation']:
        print("\n  >>> PASS: Residual separation >= full vector separation")
    else:
        ratio = resid_wb['separation'] / full_wb['separation'] if full_wb['separation'] > 1e-8 else 0
        print(f"\n  >>> {'WARN' if ratio > 0.5 else 'FAIL'}: Residual separation = {ratio:.1%} of full vector separation")

    # =========================================================================
    # Step 3: Nearest neighbors
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: NEAREST NEIGHBORS IN RESIDUAL SPACE")
    print(f"{'='*60}")

    target_nn = ["ghost", "librarian", "demon"]
    full_nn = find_nearest_neighbors(full_vecs_at_layer, target_nn, n_neighbors=8)
    resid_nn = find_nearest_neighbors(residual_vecs, target_nn, n_neighbors=8)

    for role in target_nn:
        if role not in full_nn:
            continue
        print(f"\n  {role.upper()}")
        print(f"    Full-vector neighbors:")
        for nb in full_nn[role]:
            print(f"      {nb['role']:20s}  cos={nb['cosine_similarity']:.4f}")
        print(f"    Residual neighbors:")
        for nb in resid_nn.get(role, []):
            print(f"      {nb['role']:20s}  cos={nb['cosine_similarity']:.4f}")

    # =========================================================================
    # Step 4: Clustering
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 4: CLUSTERING ANALYSIS")
    print(f"{'='*60}")

    print("\n--- Full vectors ---")
    full_clustering = run_clustering(full_vecs_at_layer, role_to_cat)
    for k, stats in full_clustering.items():
        if isinstance(stats, dict) and "adjusted_rand_index" in stats:
            print(f"  {k}: ARI={stats['adjusted_rand_index']:.4f}  silhouette={stats['silhouette_score']:.4f}")

    print("\n--- Residual vectors ---")
    resid_clustering = run_clustering(residual_vecs, role_to_cat)
    for k, stats in resid_clustering.items():
        if isinstance(stats, dict) and "adjusted_rand_index" in stats:
            print(f"  {k}: ARI={stats['adjusted_rand_index']:.4f}  silhouette={stats['silhouette_score']:.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print("PHASE 0 SUMMARY")
    print(f"{'='*60}")

    resid_frac = norm_stats['residual_fraction_mean']
    sep_ratio = resid_wb['separation'] / full_wb['separation'] if full_wb['separation'] > 1e-8 else 0

    print(f"\n  Residual fraction of full norm: {resid_frac:.1%}")
    print(f"  Residual separation / full separation: {sep_ratio:.1%}")

    if resid_frac > 0.5 and sep_ratio > 0.5:
        verdict = "PASS"
        print(f"\n  VERDICT: {verdict}")
        print("  Residuals carry substantial persona-specific information.")
        print("  Proceed with residual decomposition in main experiment.")
    elif resid_frac > 0.3 or sep_ratio > 0.3:
        verdict = "MARGINAL"
        print(f"\n  VERDICT: {verdict}")
        print("  Residuals carry some persona information but axis dominates.")
        print("  Include residual steering as exploratory; focus on role-vector steering.")
    else:
        verdict = "FAIL"
        print(f"\n  VERDICT: {verdict}")
        print("  Residuals do not carry meaningful persona-specific information.")
        print("  Pivot to raw role-vector steering only.")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "layer": layer,
            "n_roles": len(role_vectors),
            "n_categorized": len(role_to_cat),
            "verdict": verdict,
            "norm_stats": norm_stats,
            "full_vector_similarity": full_wb,
            "residual_similarity": resid_wb,
            "separation_ratio": sep_ratio,
            "clustering_full": full_clustering,
            "clustering_residual": resid_clustering,
            "nearest_neighbors_full": full_nn,
            "nearest_neighbors_residual": resid_nn,
            "target_role_norms": {
                role: {
                    "full_norm": decomp[role]["full_norm"],
                    "proj_norm": decomp[role]["proj_norm"],
                    "residual_norm": decomp[role]["residual_norm"],
                    "proj_scalar": decomp[role]["proj_scalar"],
                }
                for role in target_roles if role in decomp
            },
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
