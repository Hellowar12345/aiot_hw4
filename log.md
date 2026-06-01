# 🚀 RSP 手勢辨識系統開發與優化日誌 (Project Evolution Log)

這份文件詳細記錄了本專案從最初的基礎影像分類模型，一路上遭遇各種電腦視覺 (Computer Vision) 痛點，並透過不同演算法與架構迭代，最終進化為「超輕量級高精度座標神經網路」的完整技術歷程。

---

## 階段一：CNN 模型變形問題與影像預處理優化

專案初期，我們將傳統的 SVM 模型升級為深度學習的 EfficientNet-B0。然而在實際使用 Webcam 推論時，遭遇了手部影像擷取的挑戰。

### 痛點 1：裁切範圍過大導致背景干擾
**🗣️ User Prompt:**
> 「MediaPipe 裁切範圍太大了，把臉和背景都包進去了，EfficientNet 根本不知道要看哪裡。」
> 「能不能只抓手部的部分就好？現在背景太雜，模型有時候會把我的臉也認成石頭。」

**🤖 AI Technical Action:**
一開始的邊界框 (Bounding Box) 為了維持正方形，會向外大量擴張 Padding，導致包含過多雜訊。我們修改了裁切邏輯，改採 **最緊湊裁切 (Tight Crop)**：
1. 取出 MediaPipe 21 個座標的 `x_min, x_max, y_min, y_max`。
2. 僅保留微小的 15px 安全邊距，直接將長方形的手部區域從原圖中切下，徹底排除臉部與房間背景。

### 痛點 2：長寬比失真 (Aspect Ratio Distortion)
**🗣️ User Prompt:**
> 「我的手部裁切圖是長方形不固定大小，餵進 EfficientNet-B0 會變形。」
> 「手指如果在畫面邊緣會被拉長，這樣模型會不會誤判成剪刀？請幫我改用 letterbox 方式處理，保持原本長寬比縮放，不足的地方填黑色補成正方形。」

**🤖 AI Technical Action:**
為了適應 EfficientNet 訓練時的正方形輸入標準，且不能讓手指因強制 Resize 而被拉長或壓扁，我們手刻了 **Letterbox (信箱模式)** 演算法：
1. 取得緊湊裁切圖的當前長寬，計算等比例縮放係數 `r`。
2. 將圖片的最長邊等比例縮放至目標大小 (`224`)。
3. 計算不足的短邊差值，透過 OpenCV 的 `cv2.copyMakeBorder` 於影像上下（或左右）對稱地補上純黑色 `(0, 0, 0)` 像素。
這使得模型推論時收到的影像，永遠是比例完美且去背乾淨的 224x224 正方形。

---

## 階段二：消彌領域差異 (Domain Gap)

解決了推論階段的影像問題後，我們發現模型在攝影機前的準確率依舊不如預期，這牽涉到了機器學習中經典的 Domain Gap 問題。

**🗣️ User Prompt:**
> 「因為我發現模型在訓練集上有 99% 的準確率，但實際拿攝影機照卻一直辨識錯，是不是因為訓練集沒有黑邊的關係？」
> 「請問這個資料集的圖片風格和真實拍到的手差異大嗎？有什麼方式可以縮小 domain gap？」
> 「請幫我寫 preprocess_dataset.py，讀取 dataset/train 裡的每張圖片，對每張圖執行跟 carema.py 完全一樣的 MediaPipe 抓手部 + Letterbox 補黑邊處理...」

**🤖 AI Technical Action:**
模型在訓練時看的是「去背且沒有黑邊的原始圖片」，推論時看的卻是「帶有厚厚黑邊的 Letterbox 圖片」，這會導致模型特徵對齊失敗。
我們撰寫了 `preprocess_dataset.py`，建立**統一的預處理管線 (Pipeline Consistency)**：
1. 遍歷原始資料集中的數千張圖片。
2. 對每張圖片召喚 MediaPipe 提取骨架，執行相同的 Tight Crop + Letterbox。
3. 若遇到無法辨識手部的爛圖片則直接拋棄（順便完成資料集清洗）。
重新以處理過的新資料集訓練 EfficientNet-B0 後，模型成功達到了 **95.77%** 的驗證準確率。

---

## 階段三：多重架構對照實驗 (Ablation & Baselines)

為確保尋找到最優解，我們平行展開了不同架構的測試。

**🗣️ User Prompt:**
> 「作業要求要比較兩種不同的模型，我們能不能多訓練一個 ResNet 來跟 EfficientNet 對比看看誰比較強？」
> 「還是直接用 media pipe 好了就不要 effiecientnet」
> 「我想再 train 一個 CNN model 看看，看有沒有更輕量的方法。」

**🤖 AI Technical Action:**
1. **純數學幾何防呆版 (`camera_mediapipe_only.py`)**：完全捨棄機器學習模型。僅根據 MediaPipe 關節點，計算「指尖到手腕」與「指關節到手腕」的相對歐氏距離 (Euclidean Distance)，寫死規則判斷手指伸直的數量（0=石頭, 2=剪刀, 5=布）。此版本運算極快，適合作為 Baseline。
2. **ResNet-18 訓練 (`train_resnet.py`)**：新增了殘差網路 (ResNet-18) 的訓練腳本。讓專案同時具備 MobileNetV2、EfficientNet-B0 與 ResNet-18 的公平對照組，藉此驗證 EfficientNet 在參數量與特徵提取能力上的優勢。

---

## 階段四：終極進化 — 座標降維神經網路 (Landmark DNN)

儘管 EfficientNet 準確率極高，但 CNN 本質上仍受限於光影與膚色，而且在硬體資源有限的設備上暴露出了致命弱點。

**🗣️ User Prompt:**
> 「我把這包 EfficientNet 放到 Raspberry Pi 上面跑，結果超級卡，FPS 只有不到 5... 畫面根本不會動。」
> 「能幫我改用 Lightweight 的方式嗎？既然 MediaPipe 已經能抓到 21 個座標點了，我們能不能直接讓 model 去學這 21 個點的數字就好？完全不要處理圖片？」
> 「而且我的 CNN model 好像不太會辨別剪刀，如果只餵座標點給它，避開膚色跟光線，準確率會不會變高？」

**🤖 AI Technical Action:**
這是專案最重大的架構典範轉移 (Paradigm Shift)。我們決定**徹底捨棄圖片像素**，讓神經網路直接學習 3D 空間特徵。
1. **特徵提取與空間歸一化 (`extract_landmarks.py`)**：
   - 提取 MediaPipe 的 21 個關節點 (x, y, z 共 63 維度)。
   - **平移歸一化 (Translation Invariance)**：將「手腕 (節點 0)」設為 `(0,0,0)` 原點，所有點扣除手腕座標。
   - **縮放歸一化 (Scale Invariance)**：將所有座標除以手部最大伸展距離，將數值壓縮至 `[-1, 1]` 區間。
2. **輕量級模型訓練 (`train_landmark_mlp.py`)**：
   - 使用 PyTorch 建立了一個僅有約 **58,000 個參數** (相較 CNN 的 400 萬參數少了數十倍) 的多層感知機 (MLP)。
   - 由於輸入全是乾淨的空間幾何數字，模型訓練極快，且最終在測試集上達到了 **Accuracy 99.46% 的超高辨識率**。

---

## 階段五：雙重防呆與架構極簡化

最後我們將這個強大的 MLP 部署到攝影機上，並解決異常狀態處理與程式碼精簡。

**🗣️ User Prompt:**
> 「我剛剛測試的 model 他沒有 other 這個選項，就是我要剪刀石頭布外還要 Error/Others 這個分類交作業。」
> 「如果我比個讚，他還是會硬把它分類成石頭或布，要怎麼讓他知道這是不認識的手勢？」
> 「那我這樣還需要去裁剪圖片嗎? 既然只看 21 點座標，OpenCV 那些裁切跟補黑邊的程式是不是全都可以刪掉了？」

**🤖 AI Technical Action:**
1. **嚴格信心度門檻 (High Confidence Threshold) 過濾 Error 手勢**：
   因為 MLP 是針對 0/2/5 根手指訓練的，當使用者比出 1/3/4 根手指（未見過的手勢）時，Softmax 仍可能過度自信給出 99% 預測。我們捨棄了原本的幾何防呆，直接將推論的 `CONF_THRESH` 大幅提高至 **0.95 (95%)**。
   當模型信心度低於 95% 時，直接判定為 `Other` (Error)，成功以最精簡、最原生（不依賴外部幾何規則）的方式完美捕捉作業要求的第四分類。
2. **廢除影像處理層**：
   在使用者理解「MLP 架構完全不依賴圖片像素」後，我們大膽地將程式碼中所有用於 `Tight Crop`、`Letterbox` 甚至存檔 `debug_hand.jpg` 的厚重 OpenCV 邏輯**全數刪除**。
   這使得系統進化成最純粹的極簡型態：**「攝影機擷取 ➔ MediaPipe 解析特徵座標 ➔ MLP 瞬間分類」**，在 Raspberry Pi 上達成了 204 FPS 的驚人效能與精準度的巔峰。
