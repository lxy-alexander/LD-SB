import torch


def find_projection_matrix(model, rank):
    # Compute the projection matrix P using the top-k right singular vectors
    # of the first-layer weight matrix. This corresponds to the dominant
    # feature directions learned by the model.
    W = model.get_first_layer_weights()
    U, S, V = torch.svd(W)
    V_k = V[:, :rank]
    P = V_k @ V_k.t()
    return P


def evaluate_ldsb(model, P, val_features, val_labels, config):
    # Evaluate Low-Dimensional Simplicity Bias (LD-SB) by measuring how much
    model.eval()

    P = P.to(config.device)
    P_perp = torch.eye(P.size(0), device=config.device) - P

    # Use up to 1000 samples for evaluation
    n = min(1000, len(val_features))

    # Sample two independent sets of indices for x1 and x2
    torch.manual_seed(42)  # Ensures reproducibility
    idx1 = torch.randperm(len(val_features))[:n]
    idx2 = torch.randperm(len(val_features))[:n]

    x1 = val_features[idx1].to(config.device)
    x2 = val_features[idx2].to(config.device)

    # Mixed sample:
    # - P(x1) provides the "signal" subspace
    # - P_perp(x2) provides the "orthogonal" subspace
    x_mix = (P @ x1.t()).t() + (P_perp @ x2.t()).t()

    # Compute predictions
    with torch.no_grad():
        logits1 = model(x1)
        logits2 = model(x2)
        logits_mix = model(x_mix)

        pred1 = logits1.argmax(1)
        pred2 = logits2.argmax(1)
        pred_mix = logits_mix.argmax(1)

    # P_perp_pC: How often does prediction change when replacing P-component?
    # Higher values indicate stronger reliance on the P-subspace.
    p_perp_pc = (pred_mix != pred1).float().mean().item() * 100

    # P_pC: How often is the mixed prediction different from x2 (which contributed P_perp)?
    # Higher values indicate the model ignores P_perp and relies mainly on P.
    p_pc = (pred_mix != pred2).float().mean().item() * 100

    return {
        "P_perp_pC": p_perp_pc,
        "P_pC": p_pc,
        "rank_P": torch.linalg.matrix_rank(P).item(),
    }
