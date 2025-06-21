import torch

def evaluate_at_k(rbm, train_tensor, test_tensor, k=10):
    rbm.eval()
    with torch.no_grad():
        p_h, _ = rbm.sample_h(train_tensor)
        p_v, _ = rbm.sample_v(p_h)

    num_users, num_items = train_tensor.shape
    precisions = []
    average_precisions = []
    ndcgs = []

    for user in range(num_users):
        train_row = train_tensor[user]
        test_row = test_tensor[user]

        held_out = torch.nonzero(test_row).flatten()
        if held_out.numel() == 0:
            continue

        user_scores = p_v[user].clone()
        user_scores[train_row == 1] = -1e6  # mask out already-liked

        topk = torch.topk(user_scores, k=k).indices
        hits = (test_row[topk] == 1).float()

        # Precision@K
        precision = hits.sum().item() / k
        precisions.append(precision)

        # MAP@K
        if hits.sum() > 0:
            hit_ranks = torch.nonzero(hits).flatten() + 1  # positions start at 1
            precisions_at_hits = torch.cumsum(hits, dim=0)[hit_ranks - 1] / hit_ranks.float()
            average_precision = precisions_at_hits.mean().item()
        else:
            average_precision = 0.0
        average_precisions.append(average_precision)

        # NDCG@K
        device = hits.device  # ensure consistent device
        gains = hits / torch.log2(torch.arange(2, k + 2, device=device).float())
        dcg = gains.sum().item()
        ideal_hits = torch.ones(min(len(held_out), k), device=device)
        ideal_gains = ideal_hits / torch.log2(torch.arange(2, len(ideal_hits) + 2, device=device).float())
        idcg = ideal_gains.sum().item()
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)

    mean_precision = float(sum(precisions) / len(precisions)) if precisions else 0.0
    mean_map = float(sum(average_precisions) / len(average_precisions)) if average_precisions else 0.0
    mean_ndcg = float(sum(ndcgs) / len(ndcgs)) if ndcgs else 0.0

    return mean_precision, mean_map, mean_ndcg
