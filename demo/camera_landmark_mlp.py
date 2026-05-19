"""
demo/camera_landmark_mlp.py
----------------------------
用 MediaPipe 抓取即時關節點，
送進訓練好的 MLP 分類石頭剪刀布。

執行方式:
    python demo/camera_landmark_mlp.py

需要:
    landmark_data/landmark_mlp.pth
    landmark_data/label_encoder.json
    demo/hand_landmarker.task
"""

import os
import sys
import json
import time
import collections
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn

# ── 路徑設定 ──────────────────────────────────
TASK_PATH    = os.path.join('demo', 'hand_landmarker.task')
MODEL_PATH   = os.path.join('landmark_data', 'landmark_mlp.pth')
LE_PATH      = os.path.join('landmark_data', 'label_encoder.json')

# ── 推論設定 ──────────────────────────────────
CONF_THRESH   = 0.70    # 低於此信心值顯示 "Other"
SMOOTH_FRAMES = 7       # 投票平滑幀數 (降低閃爍)
DEVICE        = 'cpu'   # 攝影機即時推論用 CPU 即可

# ── 顏色 (BGR) ─────────────────────────────────
COLOR = {
    'rock':     (0, 100, 255),   # 橘紅
    'scissors': (0, 200, 100),   # 綠
    'paper':    (255, 180, 0),   # 藍
    'other':    (120, 120, 120), # 灰
}
# ─────────────────────────────────────────────


class LandmarkMLP(nn.Module):
    """與 train_landmark_mlp.py 完全一致的架構"""
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


def normalize_landmarks(hand_landmarks):
    """與 extract_landmarks.py 完全一致的正規化函式"""
    wrist = hand_landmarks[0]
    ref_x, ref_y, ref_z = wrist.x, wrist.y, wrist.z
    mid_mcp = hand_landmarks[9]
    scale = ((mid_mcp.x - ref_x)**2 +
             (mid_mcp.y - ref_y)**2 +
             (mid_mcp.z - ref_z)**2) ** 0.5 + 1e-6

    coords = []
    for lm in hand_landmarks:
        coords.append((lm.x - ref_x) / scale)
        coords.append((lm.y - ref_y) / scale)
        coords.append((lm.z - ref_z) / scale)
    return np.array(coords, dtype=np.float32)


def load_model(model_path, le_path):
    with open(le_path) as f:
        classes = json.load(f)['classes']
    num_classes = len(classes)
    model = LandmarkMLP(63, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    print(f"✅ 模型載入完成  類別: {classes}")
    return model, classes


def draw_landmarks_on_frame(frame, hand_landmarks, h, w):
    """在畫面上畫出 21 個關節點與連線"""
    # MediaPipe 手部連線定義
    connections = [
        (0,1),(1,2),(2,3),(3,4),          # 拇指
        (0,5),(5,6),(6,7),(7,8),           # 食指
        (0,9),(9,10),(10,11),(11,12),      # 中指
        (0,13),(13,14),(14,15),(15,16),    # 無名指
        (0,17),(17,18),(18,19),(19,20),    # 小指
        (5,9),(9,13),(13,17),              # 掌心
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (200, 200, 200), 1, cv2.LINE_AA)
    for i, (px, py) in enumerate(pts):
        r = 5 if i == 0 else 3
        cv2.circle(frame, (px, py), r, (0, 255, 0), -1, cv2.LINE_AA)


def main():
    # ── 檢查檔案 ──
    for path in [TASK_PATH, MODEL_PATH, LE_PATH]:
        if not os.path.exists(path):
            print(f"❌ 找不到: {path}")
            sys.exit(1)

    # ── 載入 MediaPipe ──
    print("⏳ 載入 MediaPipe...")
    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # ── 載入 MLP ──
    model, classes = load_model(MODEL_PATH, LE_PATH)

    # ── 開啟攝影機 ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 無法開啟攝影機")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    softmax      = nn.Softmax(dim=1)
    vote_buffer  = collections.deque(maxlen=SMOOTH_FRAMES)
    fps_times    = collections.deque(maxlen=30)

    print("\n🎥 攝影機啟動！按 Q 離開。")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # 左右鏡像，更直覺
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        label_text = "No Hand"
        conf_text  = ""
        box_color  = (80, 80, 80)

        if result.hand_landmarks:
            hand_landmarks = result.hand_landmarks[0]

            # 畫關節點
            draw_landmarks_on_frame(frame, hand_landmarks, h, w)

            # 正規化 → 推論
            coords = normalize_landmarks(hand_landmarks)
            x_tensor = torch.from_numpy(coords).unsqueeze(0)  # (1, 63)
            with torch.no_grad():
                logits = model(x_tensor)
                probs  = softmax(logits)[0]

            max_prob  = probs.max().item()
            pred_idx  = probs.argmax().item()
            pred_name = classes[pred_idx]

            # 加入投票 buffer
            if max_prob >= CONF_THRESH:
                vote_buffer.append(pred_name)
            else:
                vote_buffer.append('other')

            # 多數決投票
            if vote_buffer:
                vote_result = collections.Counter(vote_buffer).most_common(1)[0][0]
            else:
                vote_result = 'other'

            label_text = vote_result.upper()
            conf_text  = f"{max_prob:.0%}"
            box_color  = COLOR.get(vote_result, COLOR['other'])

            # 畫 Bounding Box
            xs = [int(lm.x * w) for lm in hand_landmarks]
            ys = [int(lm.y * h) for lm in hand_landmarks]
            pad = 20
            x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
            x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

            # 顯示各類機率 (右側小字)
            for i, cls_name in enumerate(classes):
                bar_w = int(probs[i].item() * 120)
                cy = 30 + i * 25
                cv2.rectangle(frame, (w - 140, cy), (w - 140 + bar_w, cy + 18),
                               COLOR.get(cls_name, (150, 150, 150)), -1)
                cv2.putText(frame, f"{cls_name} {probs[i]:.0%}",
                            (w - 138, cy + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # ── HUD ──
        # 主標籤
        cv2.putText(frame, label_text,
                    (20, 55), cv2.FONT_HERSHEY_DUPLEX, 1.8, box_color, 3, cv2.LINE_AA)
        if conf_text:
            cv2.putText(frame, conf_text,
                        (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

        # FPS
        fps_times.append(time.time() - t0)
        fps = 1.0 / (sum(fps_times) / len(fps_times))
        cv2.putText(frame, f"FPS: {fps:.1f}",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # 模式標籤
        cv2.putText(frame, "Landmark MLP",
                    (w - 140, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('RSP - Landmark MLP', frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 結束。")


if __name__ == '__main__':
    main()
