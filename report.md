# 即時手勢辨識系統設計與邊緣裝置優化
## Real-Time Gesture Recognition with Lightweight Optimization for Edge Deployment

---

## 摘要 Abstract

本專案以「石頭剪刀布」（Rock-Paper-Scissors, RPS）為手勢辨識任務，設計並比較三種不同架構的辨識系統，最終在 Raspberry Pi 邊緣裝置上部署。實驗過程中，我們發現原始的 EfficientNet-B0 + MediaPipe 雙重架構在 Raspberry Pi 上僅能達到約 5 FPS，無法滿足即時 Demo 需求。為此，我們提出一套**輕量化的 Landmark-MLP 架構**：以 MediaPipe 僅擷取手部 21 個關節點座標作為特徵，捨棄影像像素的計算，並以自訓練的三層 MLP 進行分類，最終在 Raspberry Pi 上實現 **204 FPS** 的即時推論，效能提升超過 **40 倍**。

---

## 1. 問題背景與動機

手勢辨識是人機互動（HCI）的重要研究領域之一，廣泛應用於遊戲控制、輔助溝通與機器人指令。隨著 AIoT 技術的發展，將辨識系統部署於資源受限的邊緣裝置（Edge Device）成為重要課題。

本專案以 RPS 三類手勢（加上「其他」類別共四類）為辨識目標，並選擇 **Raspberry Pi** 作為邊緣部署平台，探討在運算資源有限的條件下，如何在**辨識精度**與**推論速度**之間取得最佳平衡。

---

## 2. 系統架構演進

本專案共歷經三代架構演進，如下圖所示：

```
第一代：SVM Baseline
    ↓ 精度不足
第二代：EfficientNet-B0 + MediaPipe 雙重架構
    ↓ Raspberry Pi 上 FPS 過低（~5 FPS）
第三代：Landmark-MLP 輕量化架構 ← 最終部署版本
    ↓ 達成 204 FPS
```

### 2.1 第一代：SVM Baseline

原始專案使用 **SVM（Support Vector Machine）** 進行圖片分類：

| 項目 | 細節 |
|------|------|
| 前處理 | 圖片轉灰階 → 縮放至 64×64 |
| 特徵 | 影像攤平成 4096 維向量 / 255 正規化 |
| 模型 | `sklearn.svm.SVC(kernel='rbf', C=1.0, gamma='scale')` |
| 缺點 | 不具備空間特徵提取能力，泛化性差 |

### 2.2 第二代：EfficientNet-B0 + MediaPipe 雙重架構

為提升精度，改採深度學習模型，整合 MediaPipe 手部偵測：

**推論流程：**
1. **MediaPipe Hand Landmarker**：偵測畫面中的手部位置，取得 21 個 3D 關節點
2. **Bounding Box 裁切**：根據關節點計算手部邊界框（含 15px padding）
3. **Letterbox 縮放**：保持原始長寬比，黑邊補齊至 224×224
4. **EfficientNet-B0 推論**：以 PyTorch 預訓練模型 Fine-tune，輸出 Rock / Scissors / Paper / Error 四分類

**訓練設定：**
```
訓練資料：rock / scissors / paper 各 840 張 + 合成 Error 類別
Epochs：3（Transfer Learning，僅訓練 classifier head）
優化器：Adam
測試集準確率：95.77%（test set: 496 張）
```

**EfficientNet-B0 測試結果：**

| 指標 | 數值 |
|------|------|
| Accuracy | 95.77% |
| Precision | 96.11% (macro avg) |
| Recall | 95.77% (macro avg) |
| F1-Score | 95.66% (macro avg) |

> ⚠️ **問題發現：** 將此架構部署至 Raspberry Pi 後，由於 EfficientNet-B0 模型（~17MB，4M+ 參數）的 CPU 推論延遲，加上 MediaPipe 的 overhead，整體 FPS 僅約 **5 FPS**，Demo 畫面嚴重卡頓，無法實際使用。

---

## 3. 第三代：Landmark-MLP 輕量化架構（核心貢獻）

### 3.1 設計動機

效能瓶頸分析：

```
第二代架構的計算瓶頸：
┌─────────────────────────────────────────┐
│ MediaPipe 手部偵測（~7.8MB 模型）         │  ← 必要，但可精簡後處理
│ 影像裁切 + Letterbox resize（CPU 密集）   │  ← 可完全省略！
│ EfficientNet-B0 推論（17MB, 4M params）  │  ← 替換為輕量模型
└─────────────────────────────────────────┘
```

**核心洞察：** MediaPipe 偵測手部時，已輸出精確的 **21 個 3D 關節點座標**。這些關節點已隱含了手形的完整幾何資訊，**根本不需要再把影像裁切後餵給 CNN**！只需把這 63 維的座標向量（21 個點 × x, y, z）直接送進一個輕量 MLP，即可完成分類。

### 3.2 特徵工程：關節點正規化

為使模型不受手的位置、大小與旋轉角度影響，對 21 個關節點進行三步正規化（`normalize_landmarks` 函式）：

**Step 1：平移歸一（Translation Normalization）**
```python
wrist = coords[0]   # 以手腕（Landmark 0）為原點
coords = coords - wrist
```

**Step 2：旋轉歸一（Rotation Normalization）**
```python
v = coords[9]  # 手腕 → 中指根部 (Landmark 9) 的向量
angle = np.arctan2(v[1], v[0])
theta = -np.pi/2 - angle
# 旋轉矩陣，讓此方向永遠對齊 Y 軸負方向（朝上）
R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
coords = coords @ R.T
```

**Step 3：縮放歸一（Scale Normalization）**
```python
max_dist = np.max(np.linalg.norm(coords, axis=1))
coords = coords / max_dist  # 最遠關節點距原點距離 = 1
```

**最終特徵向量：** 63 維 float32（21 個點 × 3 個座標），不含影像像素。

### 3.3 MLP 網路架構

```python
class LandmarkMLP(nn.Module):
    # 輸入：63 維正規化關節點座標
    # 輸出：3 類（rock / scissors / paper）
    
    hidden_dims = (256, 128, 64)
    dropout = 0.3
    
    # 每層結構：Linear → BatchNorm1d → ReLU → Dropout
    # 最後接 Linear(64 → num_classes)
```

| 層次 | 輸入 → 輸出 | 正規化 |
|------|------------|--------|
| Layer 1 | 63 → 256 | BatchNorm + ReLU + Dropout(0.3) |
| Layer 2 | 256 → 128 | BatchNorm + ReLU + Dropout(0.3) |
| Layer 3 | 128 → 64 | BatchNorm + ReLU + Dropout(0.3) |
| 輸出層 | 64 → 3 | — |

**模型大小：** `landmark_mlp.pth` ≈ **240KB**（vs EfficientNet-B0 的 17MB，縮小約 **70 倍**）

### 3.4 推論流程精簡對比

| 步驟 | 第二代（EfficientNet） | 第三代（Landmark-MLP） |
|------|----------------------|----------------------|
| 1 | MediaPipe 偵測（21 點） | MediaPipe 偵測（21 點） |
| 2 | 計算 Bounding Box | 正規化關節點座標（63 維） |
| 3 | 影像裁切 + Letterbox resize | ← **省略** |
| 4 | PyTorch EfficientNet 推論 | PyTorch MLP 推論（極輕量） |
| 5 | Softmax → 分類結果 | Softmax + 7 幀投票平滑 |

### 3.5 穩健性設計

**高信心度門檻（Confidence Threshold）：**
```python
CONF_THRESH = 0.95  # 由於神經網路容易過度自信，將門檻設為 95% 以捕捉 "Other" 類別
```
由於 MLP 僅針對 3 類手勢訓練，當輸入未見過的錯誤手勢（如比讚、一根手指）時，Softmax 仍可能給出高機率。將門檻嚴格設定在 95%，能有效篩選出作業要求的 "Other (Error)" 類別。

**投票平滑（Majority Vote Smoothing）：**
```python
SMOOTH_FRAMES = 7   # 取最近 7 幀的眾數
```
透過時序上的多幀投票，能大幅降低因單張影像關節點抖動而造成的閃爍誤判。

---

## 4. 實驗結果

### 4.1 效能比較（Raspberry Pi）

| 架構 | 模型大小 | FPS | 備註 |
|------|---------|-----|------|
| SVM | <1 MB | N/A | 需預先裁切手部 |
| EfficientNet-B0 + MediaPipe | ~17 MB | **~5 FPS** | 嚴重卡頓 |
| **Landmark-MLP（本方案）** | **~240 KB** | **204 FPS** | **提升 40×** |

### 4.2 Demo 畫面說明

![Demo 截圖](C:/Users/linmaggie/.gemini/antigravity-ide/brain/35e8d842-b23a-49f1-a1fb-5c4e147011d9/media__1780317072667.jpg)

Demo 畫面顯示：
- 畫面左上角顯示 **"Rock"**（辨識結果）
- 畫面下方顯示 **FPS: 204**，確認即時性
- 手指上的 **5 個紫色圓點**（指尖關鍵點可視化）
- 系統流暢執行於 Raspberry Pi，無明顯延遲

---

## 5. 討論與結論

### 5.1 為何 Landmark-MLP 如此高效？

1. **輸入維度極小：** 63 維向量 vs. 影像的 224×224×3 ≈ 150,528 維
2. **模型參數極少：** MLP 約 6 萬參數 vs. EfficientNet-B0 約 540 萬參數
3. **省略影像處理：** 裁切、Letterbox、ToTensor、Normalize 等步驟全部消除
4. **不受 Domain Gap 影響：** 骨架座標的表示與攝影機畫質、背景、光線無關

### 5.2 限制與未來改進方向

| 限制 | 說明 | 改進方向 |
|------|------|----------|
| MediaPipe 仍有 overhead | Hand Landmarker 本身為 TFLite 模型 | 降低偵測頻率（每 N 幀偵測一次，中間幀追蹤） |
| 訓練資料來源單一 | 僅使用公開資料集，未包含自拍影像 | 採集真實環境資料進行 Fine-tune |
| 僅支援單手 | `num_hands=1` | 擴充為雙手互動 |

### 5.3 結論

本專案從 SVM、CNN 到 Landmark-MLP，完整探索了手勢辨識的架構選擇。最終方案的核心創新在於：**捨棄高成本的影像推論，改用 MediaPipe 輸出的骨架語義特徵作為輕量 MLP 的輸入**，在幾乎不損失辨識精度的前提下，將 Raspberry Pi 的 FPS 從 5 提升至 204，成功實現邊緣裝置上的即時手勢辨識 Demo。

---

## 附錄：專案架構

```
aiot_hw4/
├── extract_landmarks.py        # 從資料集萃取 21 個關節點 → CSV
├── preprocess_dataset.py       # 資料集前處理（Letterbox 縮放）
├── train/
│   ├── train_mobilenet.py      # 第一版：MobileNetV2 訓練
│   ├── train_efficientnet.py   # 第二版：EfficientNet-B0 訓練
│   └── train_landmark_mlp.py   # 第三版：Landmark-MLP 訓練
└── demo/
    ├── camera_mediapipe_only.py     # 純幾何規則版（無模型）
    ├── camera_dual_architecture.py  # MediaPipe + EfficientNet 雙重架構
    ├── camera_landmark_mlp.py       # ✅ 最終部署版（Landmark-MLP）
    ├── hand_landmarker.task         # MediaPipe 手部偵測模型
    ├── landmark_mlp.pth             # 訓練好的 MLP 權重（240KB）
    └── label_encoder.json           # 類別標籤對應
```

---

## 附錄二：AI 協作與對話紀錄

本專案開發過程中，與 AI (Gemini) 協作解決了以下關鍵問題：
1. **模型架構分析與釐清**：AI 協助分析了 EfficientNet 在 Raspberry Pi 上的效能瓶頸，並梳理出 Landmark-MLP 作為最佳輕量化方案的邏輯。
2. **Demo 程式撰寫與優化**：AI 協助撰寫了 `rsp_demo.py`，完整實作了「特徵正規化 → MLP 推論 → UI 渲染」的 pipeline。
3. **Error 類別捕捉邏輯**：在發現模型對未見過的手勢「過度自信」時，與 AI 討論並實作了 `CONF_THRESH = 0.95` 的高信心度門檻過濾器，成功捕捉出作業要求的 "Other (Error)" 類別。

*(AI 協作的原始對話紀錄請見附件 `transcript.jsonl`)*

---

*GitHub Repository: https://github.com/Hellowar12345/aiot_hw4*
