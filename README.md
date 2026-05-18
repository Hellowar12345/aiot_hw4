# RSP Gesture Recognition (Testing Repo)

這是一個用來測試「石頭剪刀布」即時影像辨識的專案。
目前有兩種不同的實作架構，想請你 clone 下來幫忙跑跑看，比較一下哪一種比較穩定，順便幫忙抓 Bug！

## 🛠️ 環境安裝

在開始之前，請先安裝需要的套件：
```bash
pip install torch torchvision scikit-learn pillow numpy
pip install mediapipe opencv-python
```

## 🎮 測試項目一：純 MediaPipe 幾何計算版

這個版本**沒有**用到深度學習模型，完全靠 MediaPipe 抓出關節點，用「關節到手腕的距離」寫死規則來判斷 0, 2, 5 根手指。

**執行指令：**
```bash
python demo/camera_mediapipe_only.py
```
👉 **測試重點：**
- 隨便轉動手腕（打橫、朝下），看看會不會誤判。
- 比出 1, 3, 4 根手指等怪異手勢，確認是否會正確顯示 `Other`。

## 🎮 測試項目二：MediaPipe + EfficientNet-B0 雙重架構版

這個版本會先用 MediaPipe 抓出手部，把手部「緊湊裁切 + 補黑邊 (Letterbox)」變成 224x224 正方形，然後送進我訓練好的 `EfficientNet-B0` 模型來判斷。

**執行指令：**
```bash
python demo/camera_dual_architecture.py
```
👉 **測試重點：**
- 看看模型會不會因為你的房間背景或燈光而猜錯。
- 試著比出 1, 3, 4 根手指，看看 EfficientNet 會把它誤認成什麼，或是能正確判斷出 `Error`。
- 如果你有開這個程式，資料夾裡會即時生成一張 `debug_hand.jpg`，你可以打開來看看模型實際接收到的正方形裁切圖片長怎樣。

---
跑完之後再跟我說你覺得哪個版本比較好，或是遇到什麼報錯！感謝！
