import os, torch, shutil
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from dataset import load_dataset, get_dataloaders, material_names, quality_names
from model import build_model
from utils import save_checkpoint, plot_confusion_matrices, plot_epoch_accuracy, print_file_creation

# --- Config ---
CSV_PATH   = "/content/drive/MyDrive/data/streetSurfaceVis_v1_0.csv"
IMG_DIR    = "/content/drive/MyDrive/data/s_1024"
BATCH_SIZE = 8
EPOCHS     = 20
LR         = 3e-4
PATIENCE   = 5
NUM_MATERIALS, NUM_QUALITIES = 5, 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data ---
dataset = load_dataset(CSV_PATH, IMG_DIR)
train_dl, val_dl = get_dataloaders(dataset, BATCH_SIZE)

# --- Model ---
features, mat_head, qual_head = build_model(NUM_MATERIALS, NUM_QUALITIES, device)
opt = torch.optim.AdamW(list(features.parameters()) +
                        list(mat_head.parameters()) +
                        list(qual_head.parameters()), lr=LR)
lossfn = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
scaler = GradScaler()

# --- Training ---
best_metric, wait = -1.0, 0
epoch_metrics = []

for epoch in range(1, EPOCHS+1):
    # Train
    features.train(); mat_head.train(); qual_head.train()
    train_loss = 0.0
    for imgs, (mat_y, qual_y) in train_dl:
        imgs, mat_y, qual_y = imgs.to(device), mat_y.to(device), qual_y.to(device)
        with autocast():
            feats = features(imgs).flatten(1)
            m_logits = mat_head(feats)
            q_logits = qual_head(feats)
            loss = lossfn(m_logits, mat_y) + lossfn(q_logits, qual_y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        train_loss += loss.item()

    # Validate
    features.eval(); mat_head.eval(); qual_head.eval()
    val_loss = 0.0; mat_ok = qual_ok = n = 0
    with torch.no_grad():
        for imgs, (mat_y, qual_y) in val_dl:
            imgs, mat_y, qual_y = imgs.to(device), mat_y.to(device), qual_y.to(device)
            with autocast():
                feats = features(imgs).flatten(1)
                m_logits, q_logits = mat_head(feats), qual_head(feats)
                vloss = lossfn(m_logits, mat_y) + lossfn(q_logits, qual_y)
            val_loss += vloss.item()
            mat_ok += (m_logits.argmax(1) == mat_y).sum().item()
            qual_ok += (q_logits.argmax(1) == qual_y).sum().item()
            n += mat_y.size(0)

    mat_acc, qual_acc = mat_ok/n, qual_ok/n
    metric = 0.5 * (mat_acc + qual_acc)
    epoch_metrics.append(metric)
    scheduler.step(metric)

    # Save best
    if metric > best_metric:
        best_metric, wait = metric, 0
        save_checkpoint(features, mat_head, qual_head, "best_model.pt")
        flag = "✅ best updated"
    else:
        wait += 1; flag = ""

    print(f"Epoch {epoch:02d}/{EPOCHS} | train_loss={train_loss:.3f}  val_loss={val_loss:.3f}  "
          f"mat_acc={mat_acc:.3f}  qual_acc={qual_acc:.3f}  avg_acc={metric:.3f}{flag}")

    if wait >= PATIENCE:
        print("⏹️ Early stopping")
        break

# --- Save to Drive ---
os.makedirs("/content/drive/MyDrive/saved_models/", exist_ok=True)
shutil.copy("best_model.pt", "/content/drive/MyDrive/saved_models/best_model.pt")

# --- Confusion matrices ---
cm_mat  = torch.zeros(NUM_MATERIALS, NUM_MATERIALS, dtype=torch.int64)
cm_qual = torch.zeros(NUM_QUALITIES, NUM_QUALITIES, dtype=torch.int64)

features.eval(); mat_head.eval(); qual_head.eval()
with torch.no_grad():
    for imgs, (mat_y, qual_y) in val_dl:
        imgs, mat_y, qual_y = imgs.to(device), mat_y.to(device), qual_y.to(device)
        feats = features(imgs).flatten(1)
        m_logits, q_logits = mat_head(feats), qual_head(feats)
        m_pred, q_pred = m_logits.argmax(1), q_logits.argmax(1)
        for t, p in zip(mat_y.view(-1), m_pred.view(-1)):
            cm_mat[t.long(), p.long()] += 1
        for t, p in zip(qual_y.view(-1), q_pred.view(-1)):
            cm_qual[t.long(), q_pred.long()] += 1

plot_confusion_matrices(cm_mat, cm_qual, material_names, quality_names)
plot_epoch_accuracy(epoch_metrics)
print_file_creation("best_model.pt")
