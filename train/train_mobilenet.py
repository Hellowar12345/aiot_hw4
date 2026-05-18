"""
train_mobilenet.py
==================
使用 PyTorch + MobileNetV2 (Transfer Learning) 訓練石頭剪刀布 + Error 四分類模型
- 訓練資料：dataset/train/ (rock, paper, scissors)
- 測試資料：dataset/test/  (rock, paper, scissors)
- Error 類別：自動合成雜訊影像補齊第四類
- 輸出指標：Accuracy、Precision、Recall、F1-score (macro)
- 訓練完畢後將模型存至 demo/rps_mobilenet.pth
"""

import sys
import io
# 強制 UTF-8 輸出（解決 Windows CP950 編碼問題）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ─────────────────────────────────────────────
# 0. 基本設定
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

CLASSES      = ['rock', 'paper', 'scissors', 'error']
NUM_CLASSES  = 4
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_EPOCHS   = 3
LR           = 1e-4
DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(BASE_DIR, 'dataset', 'train')
TEST_DIR  = os.path.join(BASE_DIR, 'dataset', 'test')
DEMO_DIR  = os.path.join(BASE_DIR, 'demo')

print("=" * 55)
print("  MobileNetV2 Rock-Paper-Scissors Training")
print("=" * 55)
print(f"  Device    : {DEVICE}")
print(f"  Train dir : {TRAIN_DIR}")
print(f"  Test dir  : {TEST_DIR}")
print(f"  Epochs    : {NUM_EPOCHS}  |  LR: {LR}  |  Batch: {BATCH_SIZE}")
print("=" * 55)
print()


# ─────────────────────────────────────────────
# 1. 資料集類別
# ─────────────────────────────────────────────
class FolderDataset(Dataset):
    VALID_EXT = {'.png', '.jpg', '.jpeg'}

    def __init__(self, root: str, class_list: list, transform=None):
        self.samples   = []
        self.transform = transform

        for idx, cls in enumerate(class_list):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                print(f"  [WARN] Folder not found: {cls_dir}, skipping.")
                continue
            for fname in os.listdir(cls_dir):
                ext = os.path.splitext(fname)[1].lower()
                if ext in self.VALID_EXT:
                    self.samples.append((os.path.join(cls_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class SyntheticErrorDataset(Dataset):
    """合成 Error 類別：隨機雜訊 / 純色塊"""

    def __init__(self, size: int, label_idx: int, transform=None):
        self.size      = size
        self.label_idx = label_idx
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if i % 2 == 0:
            arr = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        else:
            color = np.random.randint(0, 256, 3, dtype=np.uint8)
            arr   = np.full((IMG_SIZE, IMG_SIZE, 3), color, dtype=np.uint8)
        img = Image.fromarray(arr)
        if self.transform:
            img = self.transform(img)
        return img, self.label_idx


# ─────────────────────────────────────────────
# 2. 資料前處理
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────
# 3. 建立 DataLoader
# ─────────────────────────────────────────────
print("[Step 1] Loading training data...")

rps_train   = FolderDataset(TRAIN_DIR, ['rock', 'paper', 'scissors'], train_transform)
error_count = max(100, len(rps_train) // 3)
error_train = SyntheticErrorDataset(error_count, label_idx=3, transform=train_transform)
train_dataset = ConcatDataset([rps_train, error_train])

print(f"  rock/paper/scissors samples : {len(rps_train)}")
print(f"  error synthetic samples     : {error_count}")
print(f"  total train samples         : {len(train_dataset)}")
print()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)

print("[Step 1] Loading test data...")
rps_test   = FolderDataset(TEST_DIR, ['rock', 'paper', 'scissors'], test_transform)
error_test = SyntheticErrorDataset(max(10, len(rps_test) // 3), label_idx=3,
                                   transform=test_transform)
test_dataset = ConcatDataset([rps_test, error_test])

print(f"  rock/paper/scissors samples : {len(rps_test)}")
print(f"  error synthetic samples     : {len(error_test)}")
print(f"  total test samples          : {len(test_dataset)}")
print()

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, pin_memory=False)


# ─────────────────────────────────────────────
# 4. 建立 MobileNetV2 模型
# ─────────────────────────────────────────────
print("[Step 2] Building MobileNetV2 (ImageNet pretrained)...")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# 凍結 Feature Extractor
for param in model.features.parameters():
    param.requires_grad = False

# 替換 Classifier -> 4 類輸出
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, NUM_CLASSES),
)

model = model.to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total params     : {total_params:,}")
print(f"  Trainable params : {train_params:,}")
print()


# ─────────────────────────────────────────────
# 5. Loss / Optimizer / Scheduler
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


# ─────────────────────────────────────────────
# 6. 訓練迴圈
# ─────────────────────────────────────────────
print("[Step 3] Training...\n")
best_val_acc  = 0.0
best_model_wt = None

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += imgs.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds   = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += imgs.size(0)

    val_acc = val_correct / val_total
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc  = val_acc
        best_model_wt = {k: v.clone() for k, v in model.state_dict().items()}
        flag = " <-- best"
    else:
        flag = ""

    print(f"  Epoch [{epoch:02d}/{NUM_EPOCHS}]"
          f"  Loss: {train_loss:.4f}"
          f"  Train Acc: {train_acc*100:.2f}%"
          f"  Val Acc: {val_acc*100:.2f}%{flag}")


# ─────────────────────────────────────────────
# 7. 最終評估
# ─────────────────────────────────────────────
print()
print("=" * 55)
print("  Final Evaluation (best epoch weights)")
print("=" * 55)

model.load_state_dict(best_model_wt)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs    = imgs.to(DEVICE)
        outputs = model(imgs)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

acc       = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print()
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%  (macro avg)")
print(f"  Recall    : {recall*100:.2f}%  (macro avg)")
print(f"  F1-Score  : {f1*100:.2f}%  (macro avg)")

target_names = ['Rock', 'Paper', 'Scissors', 'Error']
print()
print("  Classification Report:")
print(classification_report(all_labels, all_preds,
                             target_names=target_names, digits=4))

cm = confusion_matrix(all_labels, all_preds)
print("  Confusion Matrix (rows=actual, cols=predicted):")
header = "          " + "  ".join(f"{n:>9}" for n in target_names)
print(header)
for i, row in enumerate(cm):
    row_str = "  ".join(f"{v:>9}" for v in row)
    print(f"  {target_names[i]:>8}  {row_str}")


# ─────────────────────────────────────────────
# 8. 儲存模型
# ─────────────────────────────────────────────
os.makedirs(DEMO_DIR, exist_ok=True)
save_path = os.path.join(DEMO_DIR, 'rps_mobilenet.pth')

torch.save({
    'model_state_dict': best_model_wt,
    'classes'         : CLASSES,
    'num_classes'     : NUM_CLASSES,
    'img_size'        : IMG_SIZE,
    'val_acc'         : best_val_acc,
}, save_path)

print()
print(f"  Model saved to : {save_path}")
print(f"  Best Val Acc   : {best_val_acc*100:.2f}%")
print()
print("=" * 55)
print("  Training complete!")
print("=" * 55)
