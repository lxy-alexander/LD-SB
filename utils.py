import torch


def compute_effective_rank(W):
    """
    Compute the effective rank of a matrix using the von Neumann entropy method.

    Effective Rank = exp(Entropy(P)),
    where P contains normalized squared singular values.

    Steps:
        1. Perform SVD on W.
        2. Square singular values S -》 S^2.
        3. Normalize to form a probability distribution P.
        4. Compute entropy H(P).
        5. Effective rank = exp(H).

    Args:
        W (torch.Tensor): Weight matrix.

    Returns:
        float: Effective rank (a continuous measure of dimensionality).
    """
    # Singular value decomposition
    U, S, V = torch.svd(W)

    # Square singular values
    S2 = S**2

    # Normalize to probability distribution
    P = S2 / S2.sum()

    # Remove extremely small values to avoid log(0)
    P = P[P > 1e-10]

    # Von Neumann entropy
    H = -(P * torch.log(P)).sum()

    # Effective rank
    return torch.exp(H).item()
