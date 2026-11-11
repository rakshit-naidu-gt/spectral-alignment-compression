import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_model(model, train_ds, device, epochs=12, lr=0.05, batch_size=128, seed=0, weight_decay=0.0):
    torch.manual_seed(seed)
    model.to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    model.train()
    for ep in range(epochs):
        tot_loss, correct, n = 0.0, 0, 0
        for x, y, g in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        sched.step()
        print(f"    epoch {ep+1}/{epochs}: loss={tot_loss/n:.4f} acc={correct/n:.4f}", flush=True)
    model.eval()
    return model


@torch.no_grad()
def eval_per_group(model, ds, device, n_groups=4, batch_size=256):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    group_loss = {g: 0.0 for g in range(n_groups)}
    group_correct = {g: 0 for g in range(n_groups)}
    group_n = {g: 0 for g in range(n_groups)}
    for x, y, g in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses = F.cross_entropy(logits, y, reduction="none")
        preds = logits.argmax(1)
        for gi in range(n_groups):
            mask = (g == gi)
            if mask.sum() == 0:
                continue
            group_loss[gi] += losses[mask.to(device)].sum().item()
            group_correct[gi] += (preds[mask.to(device)] == y[mask.to(device)]).sum().item()
            group_n[gi] += mask.sum().item()
    result = {}
    for gi in range(n_groups):
        if group_n[gi] == 0:
            continue
        result[gi] = {
            "loss": group_loss[gi] / group_n[gi],
            "acc": group_correct[gi] / group_n[gi],
            "n": group_n[gi],
        }
    return result


@torch.no_grad()
def overall_eval(model, ds, device, batch_size=256):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    tot_loss, correct, n = 0.0, 0, 0
    for x, y, g in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        tot_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)
    return tot_loss / n, correct / n
