"""
Real-world replication benchmark: the standard Waterbirds dataset
(waterbird_complete95_forest2water2, Sagawa et al. 2019), mirroring the exact
4-group (label x background) spurious-correlation structure the CIFAR-10
substitute in spurious_data.py was built to approximate (Appendix: Waterbirds
replication). Loaded from a pre-downloaded HuggingFace parquet mirror
(grodino/waterbirds) since the original CodaLab-hosted download has gone
dark; images and the 95%-correlated train / group-balanced test splits are
otherwise identical to the standard benchmark.

Group ids (standard group-DRO convention, group = label*2 + place):
  0 = landbird+land  (majority, ~73% of train)
  1 = landbird+water (minority)
  2 = waterbird+land (minority, smallest group)
  3 = waterbird+water (majority)
"""
import io

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_SIZE = 128  # downsized from the standard 224 to keep CPU/MPS training tractable
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_resize_crop = transforms.Compose([
    transforms.Resize(int(IMG_SIZE * 1.14)),
    transforms.CenterCrop(IMG_SIZE),
])


class WaterbirdsDataset(Dataset):
    def __init__(self, root, split):
        df = pd.read_parquet(f"{root}/{split}.parquet", columns=["image", "label", "place"])
        n = len(df)
        imgs = np.zeros((n, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        label = df["label"].to_numpy()
        place = df["place"].to_numpy()
        for i, rec in enumerate(df["image"]):
            img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
            img = _resize_crop(img)
            imgs[i] = np.asarray(img).transpose(2, 0, 1)
        self.imgs = imgs
        self.labels = label.astype(np.int64)
        self.group = (label * 2 + place).astype(np.int64)
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        img = torch.from_numpy(self.imgs[i]).float() / 255.0
        for c in range(3):
            img[c] = (img[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        return img, self.labels[i], self.group[i]

    def group_counts(self):
        return {g: int((self.group == g).sum()) for g in range(4)}


GROUP_NAMES = {
    0: "landbird+land(maj)",
    1: "landbird+water(min)",
    2: "waterbird+land(min)",
    3: "waterbird+water(maj)",
}
