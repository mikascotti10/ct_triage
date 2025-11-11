import torch, numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

@torch.no_grad()
def eval_epoch(model, dl, device, idx_normal=0):
    model.eval()
    ys, probs_all = [], []
    for xb, yb, _ in dl:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_all.append(probs); ys += yb.numpy().tolist()
    probs_all = np.vstack(probs_all)
    pst = 1.0 - probs_all[:, idx_normal]
    yhb = (pst >= 0.5).astype(int)
    return dict(
        acc=accuracy_score(ys, yhb),
        prec=precision_score(ys, yhb, zero_division=0),
        rec=recall_score(ys, yhb),
        f1=f1_score(ys, yhb),
        auc=roc_auc_score(ys, pst),
        pos_rate_pred=float(yhb.mean()),
    )
