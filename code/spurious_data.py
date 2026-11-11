"""
Controlled spurious-correlation benchmark built on CIFAR-10 (cat vs dog), standing
in for Waterbirds/CelebA which don't fit this machine's disk budget (8GB free).

Binary task: cat (CIFAR class 3) vs dog (CIFAR class 5).
Spurious feature: a colored border (red vs blue), painted on a few border pixels.
4 groups (class x border color), with 2 "aligned" (majority, large) groups and
2 "conflicting" (minority, small) groups -- exactly Waterbirds' structure
(waterbird/landbird x water/land background), same statistical shape, no download.

Group ids: 0=cat+red (majority), 1=dog+blue (majority), 2=cat+blue (minority),
3=dog+red (minority).
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

CAT, DOG = 3, 5
BORDER_PX = 3
MINORITY_FRAC = 0.15  # 85% spurious correlation (was 95%/0.05 -- too extreme, saturated minorities)


def _paint_border(img, color):
    img = img.clone()
    if color == "red":
        c = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    else:
        c = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
    img[:, :BORDER_PX, :] = c
    img[:, -BORDER_PX:, :] = c
    img[:, :, :BORDER_PX] = c
    img[:, :, -BORDER_PX:] = c
    return img


class SpuriousCatsDogs(Dataset):
    def __init__(self, root, train, seed=0):
        base = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        imgs, labels = [], []
        for img, lbl in zip(base.data, base.targets):
            if lbl in (CAT, DOG):
                imgs.append(img)
                labels.append(0 if lbl == CAT else 1)  # 0=cat, 1=dog
        imgs = np.stack(imgs)  # (N, 32, 32, 3) uint8
        labels = np.array(labels)

        rng = np.random.default_rng(seed)
        n = len(labels)
        group = np.zeros(n, dtype=int)
        border_is_red = np.zeros(n, dtype=bool)

        for cls, (aligned_color_is_red, minority_group_id, majority_group_id) in {
            0: (True, 2, 0),   # cat: majority=red(group0), minority=blue(group2)
            1: (False, 3, 1),  # dog: majority=blue(group1), minority=red(group3)
        }.items():
            idx = np.where(labels == cls)[0]
            rng.shuffle(idx)
            n_minority = int(round(MINORITY_FRAC * len(idx)))
            minority_idx = idx[:n_minority]
            majority_idx = idx[n_minority:]
            group[majority_idx] = majority_group_id
            group[minority_idx] = minority_group_id
            border_is_red[majority_idx] = aligned_color_is_red
            border_is_red[minority_idx] = not aligned_color_is_red

        self.imgs = imgs
        self.labels = labels
        self.group = group
        self.border_is_red = border_is_red
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = torch.from_numpy(self.imgs[i]).permute(2, 0, 1).float() / 255.0
        img = _paint_border(img, "red" if self.border_is_red[i] else "blue")
        img = (img - 0.5) / 0.5
        return img, self.labels[i], self.group[i]

    def group_counts(self):
        return {g: int((self.group == g).sum()) for g in range(4)}


GROUP_NAMES = {0: "cat+red(maj)", 1: "dog+blue(maj)", 2: "cat+blue(min)", 3: "dog+red(min)"}
