# utils.py
import torch


def compute_effective_rank(W):
    U, S, V = torch.svd(W)
    S2 = S**2
    P = S2 / S2.sum()
    P = P[P > 1e-10]
    H = -(P * torch.log(P)).sum()
    return torch.exp(H).item()
