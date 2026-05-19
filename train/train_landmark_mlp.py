"""
train/train_landmark_mlp.py
----------------------------
讀取 landmark_data/train.csv 與 test.csv，
訓練一個輕量 MLP (多層感知機) 分類石頭剪刀布。

特色:
- 輸入: 63 個正規化後的 landmark 座標 (純數字，不受光線影響)
- 輸出: rock / scissors / paper / other (4 類)
- 訓練資料增強: 對 landmark 加微小隨機噪點
- 輸出模型: landmark_data/landmark_mlp.pth
           landmark_data/label_encoder.json

用法:
    python train/train_landmark_mlp.py
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ── 超參數 ────────────────────────────────────
DATA_DIR    = 'landmark_data'
SAVE_DIR    = 'landmark_data'
BATCH_SIZE  = 256
EPOCHS      = 100
LR          = 1e-3
HIDDEN_DIMS = [256, 128, 64]  # MLP 各層神經元數
DROPOUT     = 0.3
NOISE_STD   = 0.01           # 資料增強: 訓練時加入的高斯噪點強度
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
# ─────────────────────────────────────────────


class LandmarkDataset(Dataset):
    def __init__(self, csv_path, label_encoder=None, augment=False):
        df = pd.read_csv(csv_path)
        labels_raw = df['label'].values
        self.features = df.drop(columns=['label']).values.astype(np.float32)
        self.augment  = augment

        if label_encoder is None:
            self.le = LabelEncoder()
            self.le.fit(labels_raw)
        else:
            self.le = label_encoder

        self.labels = self.le.transform(labels_raw).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.features[idx].copy()
        if self.augment:
            x += np.random.normal(0, NOISE_STD, x.shape).astype(np.float32)
        return torch.from_numpy(x), self.labels[idx]


class LandmarkMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train():
    train_csv = os.path.join(DATA_DIR, 'train.csv')
    test_csv  = os.path.join(DATA_DIR, 'test.csv')

    if not os.path.exists(train_csv):
        print(f"❌ 找不到 {train_csv}，請先執行 extract_landmarks.py")
        return

    print("📦 載入資料集...")
    train_ds = LandmarkDataset(train_csv, augment=True)
    le       = train_ds.le
    num_classes = len(le.classes_)
    print(f"   類別: {list(le.classes_)}  ({num_classes} 類)")
    print(f"   訓練樣本: {len(train_ds)}")

    has_test = os.path.exists(test_csv)
    if has_test:
        test_ds = LandmarkDataset(test_csv, label_encoder=le, augment=False)
        print(f"   測試樣本: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE == 'cuda'))
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0) if has_test else None

    input_dim = 63  # 21 landmarks × 3 (x, y, z)
    model = LandmarkMLP(input_dim, num_classes, HIDDEN_DIMS, DROPOUT).to(DEVICE)
    print(f"\n🧠 模型架構:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   參數量: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)
    best_path = os.path.join(SAVE_DIR, 'landmark_mlp.pth')

    print(f"\n🚀 開始訓練 ({DEVICE}) ...")
    for epoch in range(1, EPOCHS + 1):
        # ── 訓練 ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * len(y)
            train_correct += (out.argmax(1) == y).sum().item()
        scheduler.step()

        train_loss /= len(train_ds)
        train_acc   = train_correct / len(train_ds)

        # ── 測試 ──
        test_info = ""
        if test_loader:
            model.eval()
            test_correct = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    out = model(x)
                    test_correct += (out.argmax(1) == y).sum().item()
            test_acc = test_correct / len(test_ds)
            test_info = f"  test_acc={test_acc:.4f}"

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), best_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={train_loss:.4f}  "
                  f"train_acc={train_acc:.4f}{test_info}")

    # 如果沒有 test，存最後一個 epoch
    if not test_loader:
        torch.save(model.state_dict(), best_path)

    # ── 最終評估 ──
    if test_loader:
        print(f"\n📊 最佳模型測試準確率: {best_acc:.4f}")
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model.eval()
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(DEVICE)
                pred = model(x).argmax(1).cpu().numpy()
                all_pred.extend(pred)
                all_true.extend(y.numpy())

        print("\nClassification Report:")
        print(classification_report(all_true, all_pred,
                                    target_names=le.classes_))
        print("Confusion Matrix:")
        print(confusion_matrix(all_true, all_pred))

    # ── 儲存 Label Encoder ──
    le_path = os.path.join(SAVE_DIR, 'label_encoder.json')
    with open(le_path, 'w') as f:
        json.dump({'classes': list(le.classes_)}, f, ensure_ascii=False)

    print(f"\n✅ 模型已儲存: {best_path}")
    print(f"✅ Label encoder: {le_path}")
    print("\n接下來執行: python demo/camera_landmark_mlp.py")


if __name__ == '__main__':
    train()
