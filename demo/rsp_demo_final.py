"""
rsp_demo.py
===========
即時石頭剪刀布辨識 Demo
架構：MediaPipe Hand Landmarker + 訓練好的 Landmark-MLP

四種辨識結果:
  Rock     (石頭)  — 紫色
  Paper    (布)    — 黃色
  Scissors (剪刀)  — 綠色
  Other            — 橘色

模型說明:
  - MediaPipe: 偵測 21 個 3D 手部關節點
  - Landmark-MLP: 以正規化後的 63 維關節座標作為輸入進行分類
    架構: Linear(63→256) → BN → ReLU → Dropout
        → Linear(256→128) → BN → ReLU → Dropout
        → Linear(128→64)  → BN → ReLU → Dropout
        → Linear(64→3)

需求:
  pip install mediapipe opencv-python torch

執行:
  python rsp_demo.py
  （首次執行會自動下載模型與 hand_landmarker.task）
"""

import os
import sys
import json
import math
import time
import urllib.request
import collections

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn

# ─── 自動下載設定 ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
    "landmark_mlp.pth": (
        "https://raw.githubusercontent.com/Hellowar12345/aiot_hw4/"
        "main/demo/landmark_mlp.pth"
    ),
    "label_encoder.json": (
        "https://raw.githubusercontent.com/Hellowar12345/aiot_hw4/"
        "main/demo/label_encoder.json"
    ),
}

TASK_PATH    = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_PATH   = os.path.join(BASE_DIR, "landmark_mlp.pth")
LE_PATH      = os.path.join(BASE_DIR, "label_encoder.json")

# ─── 推論設定 ─────────────────────────────────────────────────────────────────
CONF_THRESH  = 0.95      # 提高門檻：神經網路很容易過度自信，設 95% 以下都算 Other
SMOOTH_N     = 7         # 投票平滑幀數

# ─── 顏色 (BGR) ───────────────────────────────────────────────────────────────
COLORS = {
    'rock':     ( 200,  50, 200),   # 紫
    'paper':    (   0, 220, 220),   # 黃
    'scissors': (  30, 200,  30),   # 綠
    'other':    (   0, 165, 255),   # 橘
}

FINGERTIP_IDS = [4, 8, 12, 16, 20]

# ─── MLP 網路定義（必須與訓練時完全一致）────────────────────────────────────────
class LandmarkMLP(nn.Module):
    def __init__(self, input_dim, num_classes,
                 hidden_dims=(256, 128, 64), dropout=0.3):
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

# ─── 工具函式 ─────────────────────────────────────────────────────────────────

def download_file(filename, url):
    """下載單一檔案，並顯示進度條"""
    path = os.path.join(BASE_DIR, filename)
    if os.path.exists(path):
        return
    print(f"Downloading {filename} ...")

    def progress(b, bs, total):
        pct = min(b * bs / total * 100, 100)
        bar = "#" * int(pct / 5) + "-" * (20 - int(pct / 5))
        print(f"\r   [{bar}] {pct:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, path, reporthook=progress)
    print(f"\n   [OK] Saved to {path}")


def download_all():
    for fname, url in FILES.items():
        download_file(fname, url)


def normalize_landmarks(hand_landmarks) -> np.ndarray:
    """
    三步正規化（必須與訓練時的 extract_landmarks.py 完全一致）
      1. 平移：以手腕 (Landmark 0) 為原點
      2. 旋轉：讓手腕→中指根部 (Landmark 9) 方向固定朝上 (Y 軸負向)
      3. 縮放：距原點最遠的關節點 = 1
    回傳：63 維 float32 向量 [x0,y0,z0, x1,y1,z1, ...]
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks],
                      dtype=np.float64)

    # 1. 平移
    wrist  = coords[0].copy()
    coords -= wrist

    # 2. 旋轉（以 XY 平面 2D 旋轉對齊 Y 軸負向）
    v     = coords[9]
    angle = np.arctan2(v[1], v[0])
    theta = -np.pi / 2 - angle
    c, s  = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])
    coords = coords @ R.T

    # 3. 縮放
    max_d = np.max(np.linalg.norm(coords, axis=1))
    if max_d > 0:
        coords /= max_d

    return coords.flatten().astype(np.float32)


def load_model(le_path: str, model_path: str, device: str):
    """載入 label encoder 與 MLP 模型"""
    with open(le_path, "r") as f:
        data = json.load(f)
    
    # 根據你的資料夾結構 (rock, scissors, paper)
    # 通常 label encoder 照字母排序會是: 0: paper, 1: rock, 2: scissors
    # 如果辨識相反，我們可以調整這裡
    # 修正：根據測試結果，0 是 rock，1 是 paper，2 是 scissors
    idx2label = {0: "rock", 1: "paper", 2: "scissors"}
    num_classes = 3

    model = LandmarkMLP(input_dim=63, num_classes=num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, idx2label


def draw_ui(frame, label: str, conf: float, fps: int, has_hand: bool):
    """疊加辨識結果文字與 FPS"""
    h, w = frame.shape[:2]
    color = COLORS.get(label, (200, 200, 200))

    # 半透明頂部黑條
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # 結果文字
    display = label.capitalize() if has_hand else "No Hand"

    cv2.putText(frame, display, (18, 58),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, color, 2, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS: {fps}", (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (190, 190, 190), 1, cv2.LINE_AA)


def draw_fingertip_dots(frame, landmarks, color, w, h):
    """在 5 個指尖畫彩色圓點"""
    for tip_id in FINGERTIP_IDS:
        lm = landmarks[tip_id]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 13, color, -1)
        cv2.circle(frame, (cx, cy), 13, (255, 255, 255), 2)


# ─── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    # 1. 下載所需檔案
    download_all()

    # 2. 載入模型
    device = "cpu"
    print("Loading Landmark-MLP model...")
    model, idx2label = load_model(LE_PATH, MODEL_PATH, device)
    print(f"[OK] Model loaded! Classes: {idx2label}")

    # 3. 初始化 MediaPipe Hand Landmarker
    print("Initializing MediaPipe Hand Landmarker...")
    base_opts = mp_python.BaseOptions(model_asset_path=TASK_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base_opts,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(opts)
    print("[OK] Initialization complete!")

    # 4. 開啟攝影機
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機（嘗試修改 VideoCapture(0) 的數字）")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    vote_queue  = collections.deque(maxlen=SMOOTH_N)
    fps_counter = 0
    fps_display = 0
    t_fps       = time.time()

    print("[Camera] ON! Press 'q' or Esc to exit.")
    print("   Rock=Purple / Paper=Yellow / Scissors=Green / Other=Orange")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # FPS 計算
        fps_counter += 1
        elapsed = time.time() - t_fps
        if elapsed >= 0.5:
            fps_display = int(fps_counter / elapsed)
            fps_counter = 0
            t_fps = time.time()

        h, w = frame.shape[:2]

        # MediaPipe 偵測
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        has_hand = bool(result.hand_landmarks)
        label    = "other"
        conf     = 0.0

        if has_hand:
            landmarks = result.hand_landmarks[0]

            # 正規化 → MLP 推論
            feat   = normalize_landmarks(landmarks)
            tensor = torch.from_numpy(feat).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1)[0]
                pred   = int(probs.argmax())
                conf   = float(probs[pred])

            raw_label = idx2label.get(pred, "other")

            # 信心度門檻（低於閾值 → Other）
            label = raw_label if conf >= CONF_THRESH else "other"

            # 投票平滑
            vote_queue.append(label)
            label = max(set(vote_queue), key=vote_queue.count)

            # 畫指尖點
            color = COLORS.get(label, (200, 200, 200))
            draw_fingertip_dots(frame, landmarks, color, w, h)
        else:
            vote_queue.clear()

        # 疊加 UI
        draw_ui(frame, label, conf, fps_display, has_hand)

        cv2.imshow("RSP Demo - Landmark MLP", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("[Exit] Bye!")


if __name__ == "__main__":
    main()
