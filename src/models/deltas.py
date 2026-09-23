import torch
from torch.nn import functional as F


def jensen_shannon_divergence(p, q):
    """Jensen-Shannon divergence after converting embeddings to distributions."""
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    m = 0.5 * (p + q)
    return 0.5 * (
        F.kl_div(m.log(), p, reduction="none").sum(-1)
        + F.kl_div(m.log(), q, reduction="none").sum(-1)
    )


def wasserstein_distance(u_values, v_values):
    """First Wasserstein distance for equally weighted samples."""

    def compute_wasserstein(u, v):
        u_sorted, _ = torch.sort(u, dim=-1)
        v_sorted, _ = torch.sort(v, dim=-1)

        all_values = torch.cat([u_sorted, v_sorted], dim=-1)
        all_values, _ = torch.sort(all_values, dim=-1)
        deltas = torch.diff(all_values, dim=-1)

        evaluation_points = all_values[..., :-1]
        u_cdf = torch.searchsorted(u_sorted.contiguous(), evaluation_points, right=True)
        v_cdf = torch.searchsorted(v_sorted.contiguous(), evaluation_points, right=True)

        u_cdf = u_cdf.to(all_values.dtype) / u.shape[-1]
        v_cdf = v_cdf.to(all_values.dtype) / v.shape[-1]
        return torch.sum(torch.abs(u_cdf - v_cdf) * deltas, dim=-1)

    return torch.stack(
        [
            compute_wasserstein(u_values[:, i, :], v_values[:, i, :])
            for i in range(u_values.shape[1])
        ],
        dim=1,
    )


def pairwise_distance(input_embs, output_embs, p=2):
    return torch.norm(input_embs - output_embs, p=p, dim=-1)


DISTANCES = {
    "cosine": lambda input_embs, output_embs: (
        1 - F.cosine_similarity(input_embs, output_embs, dim=-1)
    )
    / 2,
    "euclidean": lambda input_embs, output_embs: pairwise_distance(
        input_embs, output_embs, p=2
    ),
    "manhattan": lambda input_embs, output_embs: pairwise_distance(
        input_embs, output_embs, p=1
    ),
    "jensen_shannon": jensen_shannon_divergence,
    "wasserstein": wasserstein_distance,
    "chebyshev": lambda input_embs, output_embs: torch.max(
        torch.abs(input_embs - output_embs), dim=-1
    ).values,
    "minkowski": lambda input_embs, output_embs: pairwise_distance(
        input_embs, output_embs, p=3
    ),
}


def f_generate_deltas(input_embs, output_embs, distance_metric="cosine"):
    if distance_metric not in DISTANCES:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    return DISTANCES[distance_metric](input_embs, output_embs)
