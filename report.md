# 即時手勢辨識系統設計與邊緣裝置優化
## Real-Time Gesture Recognition with Lightweight Optimization for Edge Deployment

---

## 摘要 Abstract

本專案以「石頭剪刀布」（Rock-Paper-Scissors, RPS）為手勢辨識任務，設計並比較多種不同架構的辨識系統，最終在 Raspberry Pi 邊緣裝置上部署。實驗過程中，我們發現深度學習模型（如 EfficientNet-B0）在實際 Demo 時遭遇嚴重的 Domain Gap 問題，導致準確率大幅下降。為此，我們提出一套**輕量化的 Landmark-MLP 架構**：捨棄像素推論，改以 MediaPipe 擷取手部 21 個關節點座標作為特徵，並以自訓練的三層 MLP 進行分類。隨後針對 Raspberry Pi 運算瓶頸進行推論管線優化，最終實現 **204 FPS** 的即時推論，準確率達 **99.46%**。

---

## 1. 問題背景與動機

手勢辨識是人機互動（HCI）的重要研究領域之一，廣泛應用於遊戲控制、輔助溝通與機器人指令。隨著 AIoT 技術的發展，將辨識系統部署於資源受限的邊緣裝置（Edge Device）成為重要課題。

本專案以 RPS 三類手勢（加上「其他」類別共四類）為辨識目標，並選擇 **Raspberry Pi** 作為邊緣部署平台，探討在運算資源有限的條件下，如何在**辨識精度**與**推論速度**之間取得最佳平衡。

---

## 2. 系統架構演進

本專案共歷經三代架構演進，如下圖所示：

```
第一代：SVM Baseline
    ↓ 精度不足，升級深度學習
第二代：EfficientNet-B0 + MediaPipe 雙重架構
    ↓ 測試集 95.77%，但實際攝影機 Demo 時 Domain Gap 嚴重
      剪刀手勢常誤判，光線變化即失效
第三代：Landmark-MLP 輕量化架構（捨棄像素，改學骨架座標）
    ↓ 準確率躍升至 99.46%，但 MediaPipe + MLP 在 Raspberry Pi 上 FPS 仍受限
第三代優化：Lightweight 管線（精簡推論流程）
    ↓ 最終達成 204 FPS ← 最終部署版本
```

### 2.1 第一代：SVM Baseline

原始專案使用 **SVM（Support Vector Machine）** 進行圖片分類：

| 項目 | 細節 |
|------|------|
| 前處理 | 圖片轉灰階 → 縮放至 64×64 |
| 特徵 | 影像攤平成 4096 維向量 / 255 正規化 |
| 模型 | `sklearn.svm.SVC(kernel='rbf', C=1.0, gamma='scale')` |
| 缺點 | 不具備空間特徵提取能力，泛化性差 |

### 2.1.5 初步探索：MobileNetV2 基準實驗

在正式選定主力模型前，我們先以 **MobileNetV2** 作為輕量級深度學習基準進行對照，原因是其參數量小（~3.4M），理論上適合未來邊緣裝置部署。

| 項目 | 細節 |
|------|------|
| 架構 | MobileNetV2（ImageNet 預訓練 + Fine-tune） |
| 訓練設定 | 凍結前段特徵層，僅訓練分類頭 |
| 測試集準確率 | ~88% |
| 缺點 | 對剪刀手勢辨識不穩定，準確率不及預期 |

由於 MobileNetV2 的準確率約在 88%，辨識剪刀手勢時尤其不穩定，因此決定進一步改用特徵提取能力更強的 EfficientNet-B0 作為主要比較模型（訓練腳本：`train/train_mobilenet.py`）。

### 2.2 第二代：EfficientNet-B0 + MediaPipe 雙重架構

為提升精度，改採深度學習模型，整合 MediaPipe 手部偵測：

**推論流程：**
1. **MediaPipe Hand Landmarker**：偵測畫面中的手部位置，取得 21 個 3D 關節點
2. **Bounding Box 裁切**：根據關節點計算手部邊界框（含 15px padding）
3. **Letterbox 縮放**：保持原始長寬比，黑邊補齊至 224×224
4. **EfficientNet-B0 推論**：以 PyTorch 預訓練模型 Fine-tune，輸出 Rock / Scissors / Paper / Other 四分類

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

> ⚠️ **問題發現：** 雖然測試集準確率達到 95.77%，但將此架構實際接上攝影機進行 Demo 時，卻遭遇嚴重的 **Domain Gap（領域差異）** 問題。訓練集為乾淨去背的手部圖片，而真實攝影機拍攝的影像帶有背景雜訊與室內光線變化，導致「剪刀」手勢常常被誤判，在光線較暗的環境下更是大幅失效。此致命缺陷促使我們徹底重新思考架構，放棄基於像素的 CNN 推論。

---

## 3. 第三代：Landmark-MLP 輕量化架構（核心貢獻）

### 3.1 設計動機

第二代架構面臨 Domain Gap 導致實際 Demo 準確率崩潰的問題。EfficientNet-B0 本質上學習的是像素的統計特徵，對光線、膚色與背景極度敏感，且難以透過更多訓練資料根治。

實驗中發現，若直接使用純幾何規則（計算手指伸直數量）判斷手勢，效果已優於 EfficientNet。這揭示了一個關鍵洞察：**手勢的辨識資訊不在像素裡，而在骨架的幾何形狀裡**。

```
第二代架構的計算瓶頸與精簡：
┌─────────────────────────────────────────┐
│ MediaPipe 手部偵測（~7.8MB 模型）         │  ← 必要，保留
│ 影像裁切 + Letterbox resize（CPU 密集）   │  ← 可完全省略！
│ EfficientNet-B0 推論（17MB, 4M params）  │  ← 替換為輕量 MLP
└─────────────────────────────────────────┘
```

**核心洞察：** MediaPipe 偵測手部時，已輸出精確的 **21 個 3D 關節點座標**。只需把這 63 維的座標向量（21 個點 × x, y, z）直接送進一個輕量 MLP，即可完成分類，**完全不需要處理任何像素**！

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

### 3.4 穩健性設計：Error 類別捕捉

MLP 僅針對 3 類手勢訓練（rock / scissors / paper），「Other」類別透過以下**雙重防線**實現：

**第一道：幾何規則過濾器**
先用幾何演算法計算手指伸直狀態，若判定手勢根本不像 RSP（例如比讚只有拇指翹起），直接判 Other，不讓 MLP 開口。

**第二道：高信心度門檻（Confidence Threshold）**
```python
CONF_THRESH = 0.95  # 由於神經網路容易過度自信，將門檻設為 95%
```
即使通過幾何檢查，若 MLP 信心度低於 95%，仍判定為 Other，避免邊界情況被強制歸類。

**投票平滑（Majority Vote Smoothing）：**
```python
SMOOTH_FRAMES = 7   # 取最近 7 幀的眾數
```
透過時序上的多幀投票，大幅降低因單張影像關節點抖動造成的閃爍誤判。

---

## 4. 部署至 Raspberry Pi 與效能優化 (Lightweight)

解決了 Domain Gap 的準確率問題後，我們將 Landmark-MLP 架構部署至 Raspberry Pi。此時發現，雖然 MLP 本身推論幾乎無延遲（僅約 6 萬參數），但 **MediaPipe Hand Landmarker 每一幀都必須在 CPU 上執行完整的手部偵測模型**。對算力有限的嵌入式設備而言，這仍造成巨大的 overhead，導致 FPS 不足，畫面卡頓。

**優化策略：Lightweight 推論管線精簡**

為提升 FPS，我們針對管線進行以下輕量化調整：
1. **降低攝影機輸入解析度**：減少 MediaPipe 每幀需要處理的像素總數。
2. **調整追蹤閾值**：優化 `min_tracking_confidence`，使追蹤更積極，減少頻繁觸發完整偵測的負擔。
3. **極簡化 UI 渲染**：移除所有非必要的畫面繪製開銷（例如不畫完整骨架網，僅保留 5 個指尖點作為視覺化）。
4. **維持 63 維幾何推論**：MediaPipe 底層仍偵測 21 個點以供 MLP 進行高精度判斷。

**優化結果：** 成功在 Raspberry Pi 上實現 **204 FPS** 的極高更新率，相較於 EfficientNet 時期提升超過 **40 倍**，且辨識準確率維持穩定。

---

## 5. 實驗結果

### 5.1 模型辨識指標比較

將本專案測試的兩種深度學習模型進行精確度評估（測試集共 369 張）：

**1. EfficientNet-B0 測試結果：**
| 指標 | 數值 |
|------|------|
| Accuracy | 95.77% |
| Precision | 96.11% |
| Recall | 95.77% |
| F1-Score | 95.66% |

**2. Landmark-MLP（本方案）測試結果：**
| 指標 | 數值 |
|------|------|
| Accuracy | **99.46%** |
| Precision | **99.00%** |
| Recall | **99.00%** |
| F1-Score | **99.00%** |

*(註：MLP 因為直接使用骨架座標特徵，排除了背景與光線干擾，因此在真實 Demo 與測試集上皆能達到近乎完美的表現。)*

### 5.2 推論效能比較（Raspberry Pi）

| 架構 | 模型大小 | FPS | 備註 |
|------|---------|-----|------|
| SVM | <1 MB | N/A | 需預先裁切手部 |
| EfficientNet-B0 + MediaPipe | ~17 MB | ~5 FPS | 影像處理極度耗時 |
| **Landmark-MLP（本方案）** | **~240 KB** | **204 FPS** | **提升 40×，部署成功** |

## 6. 討論與結論

### 6.1 為何 Landmark-MLP 如此高效？

1. **輸入維度極小：** 63 維向量 vs. 影像的 224×224×3 ≈ 150,528 維
2. **模型參數極少：** MLP 約 6 萬參數 vs. EfficientNet-B0 約 540 萬參數
3. **省略影像處理：** 裁切、Letterbox、ToTensor、Normalize 等步驟全部消除
4. **不受 Domain Gap 影響：** 骨架座標的表示與攝影機畫質、背景、光線無關

### 6.2 限制與未來改進方向

| 限制 | 說明 | 改進方向 |
|------|------|----------|
| MediaPipe 仍有 overhead | Hand Landmarker 本身為 TFLite 模型 | 降低偵測頻率（每 N 幀偵測一次，中間幀追蹤） |
| 訓練資料來源單一 | 僅使用公開資料集，未包含自拍影像 | 採集真實環境資料進行 Fine-tune |
| 僅支援單手 | `num_hands=1` | 擴充為雙手互動 |

### 6.3 結論

本專案從 SVM、CNN 到 Landmark-MLP，完整探索了手勢辨識的架構選擇。關鍵轉折在於放棄以像素為基礎的影像學習，改採 MediaPipe 骨架座標作為特徵，解決了 Domain Gap 問題並大幅提升準確率；隨後針對 Raspberry Pi 進行 Lightweight 管線優化，解決 MediaPipe overhead 造成的低 FPS 瓶頸，最終以 **204 FPS** 成功實現邊緣裝置上的流暢即時手勢辨識。

---

## 附錄：專案架構

```
aiot_hw4/
├── extract_landmarks.py        # 從資料集萃取 21 個關節點 → CSV
├── preprocess_dataset.py       # 資料集前處理（Letterbox 縮放）
├── train/
│   ├── train_mobilenet.py      # 初步基準：MobileNetV2 訓練
│   ├── train_efficientnet.py   # 第二代：EfficientNet-B0 訓練
│   └── train_landmark_mlp.py   # 第三代：Landmark-MLP 訓練
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
1. **模型架構分析與釐清**：AI 協助分析 EfficientNet 的 Domain Gap 問題，並梳理出以骨架座標取代像素作為 Landmark-MLP 輸入的核心邏輯。
2. **Raspberry Pi 效能優化**：AI 指出 MLP 結合 MediaPipe 後的效能瓶頸在於 MediaPipe overhead，並透過 Lightweight 方案大幅提升 FPS。
3. **Demo 程式撰寫與優化**：AI 協助撰寫了 `camera_landmark_mlp.py`，完整實作了「特徵正規化 → MLP 推論 → UI 渲染」的 pipeline。
4. **Error 類別捕捉邏輯**：在發現模型對未見過的手勢「過度自信」時，與 AI 討論並實作了幾何規則 + `CONF_THRESH = 0.95` 的雙重防線，成功捕捉出作業要求的 "Other (Error)" 類別。

*(AI 協作的完整歷程與對話紀錄請見附件 `log.md`)*

---

*GitHub Repository: https://github.com/Hellowar12345/aiot_hw4*
