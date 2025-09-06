import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsampler import ImbalancedDatasetSampler

# --- Label maps ---
master_class = {"asphalt":0,"concrete":1,"paving_stones":2,"unpaved":3,"sett":4}
sub_class    = {"excellent":0,"good":1,"intermediate":2,"bad":3,"very_bad":4}

material_names = list(master_class.keys())
quality_names  = list(sub_class.keys())

# --- Transforms ---
tfms = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def load_dataset(csv_path, img_dir):
    df = pd.read_csv(csv_path)
    img_paths   = [os.path.join(img_dir, f) for f in df["filepath"]]
    mat_labels  = [master_class[m] for m in df["type"]]
    qual_labels = [sub_class[q] for q in df["surface_quality"]]
    return list(zip(img_paths, mat_labels, qual_labels))

def collate(batch):
    imgs, mats, quals = [], [], []
    for path, m, q in batch:
        img = tfms(Image.open(path).convert("RGB"))
        imgs.append(img); mats.append(m); quals.append(q)
    return torch.stack(imgs), (torch.tensor(mats), torch.tensor(quals))

def get_dataloaders(dataset, batch_size):
    idx = torch.randperm(len(dataset))
    n_tr = int(0.8 * len(dataset))
    train_set = [dataset[i] for i in idx[:n_tr]]
    val_set   = [dataset[i] for i in idx[n_tr:]]

    train_sampler = ImbalancedDatasetSampler(
        train_set, callback_get_label=lambda ds: [x[1] for x in ds]
    )

    train_dl = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                          collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0)

    return train_dl, val_dl
