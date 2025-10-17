# src/losses/irm_penalty.py
import torch
import torch.nn.functional as F

def cross_entropy(logits, y):
    return F.cross_entropy(logits, y)

def irmv1_penalty(logits, y):
    """
    IRM 'scale-probe' penalty: take derivative of CE wrt a scale s at s=1.
    logits: [N, C], y: [N]
    returns scalar penalty
    """
    s = torch.tensor(1.0, requires_grad=True, device=logits.device)
    loss = F.cross_entropy(logits * s, y)
    grad = torch.autograd.grad(loss, [s], create_graph=True)[0]
    return grad.pow(2)
