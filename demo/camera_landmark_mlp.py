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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_PATH    = os.path.join(BASE_DIR, 'hand_landmarker.task')
MODEL_PATH   = os.path.join(BASE_DIR, 'landmark_mlp.pth')
LE_PATH      = os.path.join(BASE_DIR, 'label_encoder.json')

# ── 推論設定 ──────────────────────────────────
CONF_THRESH   = 0.70    # 低於此信心值顯示 "Other"
SMOOTH_FRAMES = 7       # 投票平滑幀數 (降低閃爍)
DEVICE        = 'cpu'   # 攝影機即時推論用 CPU 即可

# ── 顏色 (BGR) ─────────────────────────────────
COLOR = {
    'rock':     (0, 100, 255),   # 橘紅
    'scissors': (0, 200, 100),   # 綠
    'paper':    (255, 180, 0),   # 藍
    'other':    (0, 0, 255),     # 亮紅 (警告色)
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
    """必須與 extract_landmarks.py 完全一致的正規化函式"""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    
    # 1. 平移歸一化：以手腕 (第0點) 為原點
    wrist = coords[0]
    coords = coords - wrist
    
    # 2. 旋轉歸一化：讓手腕(0)到中指根部(9)的連線永遠固定朝上 (Y軸負向)
    v = coords[9] - coords[0]
    angle = np.arctan2(v[1], v[0])
    theta = -np.pi/2 - angle
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
    coords = coords @ R.T
    
    # 3. 縮放歸一化：找出距離原點最遠的點
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
        
    return coords.flatten().astype(np.float32)

def get_geometric_gesture(landmarks):
    """用最絕對的數學幾何判斷，直接計算哪幾根手指伸直"""
    import math
    def get_dist(lm1, lm2):
        return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)
    
    wrist = landmarks[0]
    
    # 判斷各手指是否伸展 (指尖到手腕距離 > 關節到手腕距離)
    index_up = get_dist(wrist, landmarks[8]) > get_dist(wrist, landmarks[6])
    middle_up = get_dist(wrist, landmarks[12]) > get_dist(wrist, landmarks[10])
    ring_up = get_dist(wrist, landmarks[16]) > get_dist(wrist, landmarks[14])
    pinky_up = get_dist(wrist, landmarks[20]) > get_dist(wrist, landmarks[18])
    
    # 拇指判斷比較特別 (用拇指尖到小指根部的距離)
    thumb_up = get_dist(landmarks[4], landmarks[17]) > get_dist(landmarks[3], landmarks[17])
    
    # 嚴格定義：只有食指與中指伸直，且拇指必須彎曲 (避免拇指+食指+中指被誤判) -> 就是剪刀
    if not thumb_up and index_up and middle_up and not ring_up and not pinky_up:
        return 'scissors'
        
    # 全部彎曲，包含拇指 (避免「比讚」被誤判) -> 石頭
    if not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return 'rock'
        
    # 五指全開 -> 布
    if index_up and middle_up and ring_up and pinky_up and thumb_up:
        return 'paper'
        
    return 'other'


def load_model(model_path, le_path):
    with open(le_path) as f:
        classes = json.load(f)['classes']
    
    # 如果標籤是整數 [0, 1, 2]，手動轉回正確的字串名稱
    if classes == [0, 1, 2] or classes == ["0", "1", "2"]:
        classes = ['rock', 'paper', 'scissors']
        
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
    # 預設為 1 (外接攝影機)。如果打不開，請改回 0 (筆電內建攝影機)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

            # 🚀 退回防線：讓 MLP 親自上陣！
            # 幾何數學只用來當作「防呆過濾器 (Anomaly Detector)」，判斷是否為 OTHER
            geom_gesture = get_geometric_gesture(hand_landmarks)
            
            # 如果數學計算發現這是一個正常的剪刀、石頭、布的手勢
            if geom_gesture != 'other' and max_prob >= CONF_THRESH:
                # 就完全信任 MLP 模型自己的判斷！
                vote_buffer.append(pred_name)
            else:
                # 如果手勢不合常理 (如比讚)，或者 MLP 信心度太低，才輸出 OTHER
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
