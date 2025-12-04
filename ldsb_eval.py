# ldsb_eval.py
import torch
from utils import compute_effective_rank

def find_projection_matrix(model, rank):
    W = model.get_first_layer_weights()
    U, S, V = torch.svd(W)
    V_k = V[:, :rank]
    return V_k @ V_k.t()


def evaluate_ldsb(model, P, val_features, val_labels, config):
    model.eval()

    P = P.to(config.device)
    P_perp = torch.eye(P.size(0)).to(config.device) - P

    n = min(1000, len(val_features))
    idx1 = torch.randint(0, len(val_features), (n,))
    idx2 = torch.randint(0, len(val_features), (n,))

    x1 = val_features[idx1].to(config.device)
    x2 = val_features[idx2].to(config.device)

    x_mix = (P @ x1.t()).t() + (P_perp @ x2.t()).t()

    with torch.no_grad():
        logits1 = model(x1)
        logits2 = model(x2)
        logits_mix = model(x_mix)

        pred1 = logits1.argmax(1)
        pred2 = logits2.argmax(1)
        pred_mix = logits_mix.argmax(1)

    return {
        "P_perp_pC": (pred_mix != pred1).float().mean().item() * 100,
        "P_pC": (pred_mix != pred2).float().mean().item() * 100,
        "rank_P": torch.linalg.matrix_rank(P).item()
    }
