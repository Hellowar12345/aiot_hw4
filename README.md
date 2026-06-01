# RSP Gesture Recognition (石頭剪刀布 - 即時手勢辨識系統)

本專案為 AIoT 邊緣運算手勢辨識系統。我們比較了「EfficientNet-B0 深度影像學習模型」與「Landmark-MLP 輕量化座標神經網路」兩種架構，最終實作出一套能夠在 Raspberry Pi 等邊緣裝置上達成 **高達 204 FPS** 且 **準確率 99.4%** 的即時推論系統。

## 📂 專案架構 (Project Structure)

```text
aiot_hw4/
├── extract_landmarks.py        # 從資料集萃取 21 個關節點 → CSV
├── preprocess_dataset.py       # 資料集前處理（Letterbox 縮放）
├── transcript.jsonl            # AI 協作與對話紀錄
├── report.md                   # 期末專案技術報告 (效能比較與理論)
├── log.md                      # 開發歷程與防呆機制進化日誌
├── README.md                   # 專案說明 (本文件)
├── train/
│   ├── train_efficientnet.py   # 模型一：EfficientNet-B0 訓練腳本
│   └── train_landmark_mlp.py   # 模型二：Landmark-MLP 訓練腳本
└── demo/
    ├── camera_dual_architecture.py  # 測試一：EfficientNet 雙重架構推論
    ├── rsp_demo_final.py            # ✅ 最終部署版（Landmark-MLP 超高 FPS 推論）
    ├── hand_landmarker.task         # MediaPipe 手部偵測模型
    ├── landmark_mlp.pth             # 訓練好的 MLP 權重
    └── label_encoder.json           # 類別標籤對應
```

---

## 🛠️ 環境安裝

在開始之前，請先安裝需要的套件：
```bash
pip install torch torchvision scikit-learn pillow numpy
pip install mediapipe opencv-python
```

---

## 🎮 測試項目一：EfficientNet-B0 雙重架構版

這個版本會先用 MediaPipe 抓出手部，把手部「緊湊裁切 + 補黑邊 (Letterbox)」變成 224x224 正方形，然後送進訓練好的 `EfficientNet-B0` 影像模型來判斷。

**執行指令：**
```bash
python demo/camera_dual_architecture.py
```
👉 **測試重點：**
- 影像辨識會不會因為你的房間背景或燈光而猜錯。
- 這個架構在一般電腦上運行正常，但在 Raspberry Pi 上效能瓶頸明顯（僅約 5 FPS）。

---

## 🎮 測試項目二：Landmark-MLP 終極輕量版 (推薦 ✅)

這個版本**完全捨棄了影像像素處理**，而是提取 MediaPipe 的 3D 空間關節點座標，經過正規化後輸入極輕量的 MLP 神經網路。

**執行指令：**
```bash
python demo/rsp_demo_final.py
```
👉 **測試重點：**
- 體會其超越 EfficientNet 40倍以上的超高速推論（Raspberry Pi 可達 204 FPS）。
- 試著比出 1, 3, 4 根手指或是「比讚」，看看系統是否能透過我們設計的 **95% 嚴格信心度門檻 (CONF_THRESH)**，成功將未見過的手勢捕捉並分類為 `Other` (橘色標記)。
