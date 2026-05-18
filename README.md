# AIoT HW4: Rock-Paper-Scissors (RSP) Gesture Recognition

本專案為石頭、剪刀、布 (Rock, Paper, Scissors) 以及錯誤手勢 (Error) 的即時影像辨識系統。
系統成功整合了 **MediaPipe 手部追蹤** 與 **深度學習 CNN 模型**，能在 Raspberry Pi 或一般 PC 上流暢運行。

## 🎯 系統架構設計 (System Architecture)

為了解決傳統影像辨識容易受背景雜物與手部角度干擾的問題 (Domain Gap)，本專案實作了兩種版本的攝影機推論程式：

1. **`demo/camera_mediapipe_only.py` (純 MediaPipe 幾何防呆版)**：
   - 僅透過 MediaPipe 擷取手部 21 個 3D 關節點。
   - 利用「指尖到手腕」與「指關節到手腕」的相對距離演算法，精準判斷手指是否伸直。
   - 完全不受背景、光線干擾，精準度極高且運算極快。

2. **`demo/camera_dual_architecture.py` (雙重架構版：MediaPipe + EfficientNet-B0)**：
   - 先由 MediaPipe 定位手部，進行**緊湊裁切 (Tight Crop)**。
   - 使用 **Letterbox (補黑邊)** 技術將長方形手部影像完美補齊至 `224x224` 正方形，確保長寬比不失真。
   - 最後送入預先訓練好的 `EfficientNet-B0` 進行神經網路預測。

---

## 🔬 模型選擇與訓練結果比較 (Model Comparisons)

為了找出最適合此任務的架構，本專案實作並訓練了三種不同的經典 CNN 模型進行比較。所有模型皆新增了第四類別 `Error` (透過合成雜訊/純色影像進行訓練以防呆)，並統一輸出 Accuracy, Precision, Recall, F1-score 評估指標。

| 模型架構 (Model) | 參數大小 | Accuracy | Precision | Recall | F1-Score | 模型特色與更換原因 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | 17 MB | **95.77%** | **96.11%** | **95.77%** | **95.66%** | **(最終選擇)** 原專案使用 SVM 處理拉平特徵無法泛化。考量到要在 Raspberry Pi 運行，選擇 EfficientNet-B0，其結合深度可分離卷積，參數量少且特徵擷取能力極強，效能遠超原本架構。 |
| **ResNet-18** | 43 MB | 93.15% | 93.20% | 93.15% | 93.05% | 經典的殘差網路架構，作為 Baseline 對照組。表現穩定但參數量較大。 |
| **MobileNetV2** | 9 MB | 88.20% | 88.50% | 88.20% | 88.10% | 專為邊緣運算設計，速度最快，但在此任務中準確率不及 EfficientNet。 |

---

## 🚀 如何執行 (How to Run)

### 1. 安裝相依套件
```bash
pip install torch torchvision scikit-learn pillow numpy
pip install mediapipe opencv-python
```

### 2. 即時攝影機辨識 (Demo Camera)
可用於錄製 Demo 影片，執行手勢測試 (須包含 Rock, Scissors, Paper 以及 1/3/4 指等 Error 錯誤手勢)。
```bash
# 執行雙重架構版 (展示機器學習模型能力)
python demo/camera_dual_architecture.py

# 或執行純 MediaPipe 超快防呆版
python demo/camera_mediapipe_only.py
```
*(畫面中將顯示 MediaPipe 彩色手部骨架與最終辨識分類)*

### 3. 測試模型準確率 (Test Model)
```bash
python demo/test_efficientnet.py
```
