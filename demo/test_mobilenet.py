"""
test_mobilenet.py
==================
載入訓練好的 MobileNetV2 模型，對 dataset/test/ 進行推論
並輸出 Accuracy、Precision、Recall、F1-score
用法：在 demo/ 資料夾內執行 python test_mobilenet.py
"""

import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ─────────────────────────────────────────────
# 0. 路徑設定
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR   = os.path.join(BASE_DIR, 'dataset', 'test')
MODEL_PATH = os.path.join(BASE_DIR, 'demo', 'rps_mobilenet.pth')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"{'='*55}")
print(f"  MobileNetV2 推論評估腳本")
print(f"{'='*55}")
print(f"  模型路徑：{MODEL_PATH}")
print(f"  測試目錄：{TEST_DIR}")
print(f"  裝置　　：{DEVICE}\n")

if not os.path.exists(MODEL_PATH):
    print(f"❌ 找不到模型 '{MODEL_PATH}'")
    print("   請先執行 train/train_mobilenet.py 完成訓練！")
    sys.exit(1)


# ─────────────────────────────────────────────
# 1. 載入模型設定
# ─────────────────────────────────────────────
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
CLASSES     = checkpoint['classes']          # ['rock', 'paper', 'scissors', 'error']
NUM_CLASSES = checkpoint['num_classes']      # 4
IMG_SIZE    = checkpoint['img_size']         # 224


# ─────────────────────────────────────────────
# 2. 資料集 & Transform
# ─────────────────────────────────────────────
class FolderDataset(Dataset):
    VALID_EXT = {'.png', '.jpg', '.jpeg'}

    def __init__(self, root, class_list, transform=None):
        self.samples   = []
        self.transform = transform
        for idx, cls in enumerate(class_list):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if os.path.splitext(fname)[1].lower() in self.VALID_EXT:
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
    def __init__(self, size, label_idx, transform=None):
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


test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

print("📂 載入測試集...")
rps_test   = FolderDataset(TEST_DIR, ['rock', 'paper', 'scissors'], test_transform)
error_test = SyntheticErrorDataset(max(10, len(rps_test) // 3), label_idx=3,
                                   transform=test_transform)
test_dataset = ConcatDataset([rps_test, error_test])
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
print(f"  測試樣本總數：{len(test_dataset)}\n")


# ─────────────────────────────────────────────
# 3. 重建模型並載入權重
# ─────────────────────────────────────────────
print("🔧 重建 MobileNetV2 並載入模型權重...")
model = models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, NUM_CLASSES),
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()
print("✅ 模型載入成功！\n")


# ─────────────────────────────────────────────
# 4. 推論
# ─────────────────────────────────────────────
print("⏳ 正在對測試集進行預測...")
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


# ─────────────────────────────────────────────
# 5. 輸出評估指標
# ─────────────────────────────────────────────
acc       = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
recall    = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1        = f1_score(all_labels, all_preds, average='macro', zero_division=0)

print(f"\n{'='*55}")
print("  📊 評估結果")
print(f"{'='*55}")
print(f"  Accuracy  : {acc*100:.2f}%")
print(f"  Precision : {precision*100:.2f}%  (macro avg)")
print(f"  Recall    : {recall*100:.2f}%  (macro avg)")
print(f"  F1-Score  : {f1*100:.2f}%  (macro avg)")

target_names = ['Rock', 'Paper', 'Scissors', 'Error']
print(f"\n  詳細分類報告：")
print(classification_report(all_labels, all_preds,
                             target_names=target_names, digits=4))

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("  混淆矩陣（列=實際, 欄=預測）：")
header = "          " + "  ".join(f"{n:>9}" for n in target_names)
print(header)
for i, row in enumerate(cm):
    row_str = "  ".join(f"{v:>9}" for v in row)
    print(f"  {target_names[i]:>8}  {row_str}")

print(f"\n{'='*55}\n")
