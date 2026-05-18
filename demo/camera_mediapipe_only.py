"""
carema.py
=========
100% MediaPipe 版本：完全放棄 ML 模型，
改用「不受手勢角度影響」的 3D 骨架距離演算法來判斷石頭剪刀布！
只顯示 Rock, Paper, Scissors, Other 乾淨的結果。
"""

import cv2
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
# 1. 初始化 MediaPipe (Tasks API)
# ─────────────────────────────────────────────
print("⏳ 初始化 MediaPipe...")
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def get_distance(lm1, lm2):
    """計算兩點之間的距離"""
    return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2)

def detect_gesture(landmarks):
    """
    不管手是打橫、朝下還是朝上，都用「關節相對距離」來判斷手指有沒有伸直！
    """
    fingers_up = 0
    wrist = landmarks[0]

    # 食指 (8)、中指 (12)、無名指 (16)、小指 (20)
    # 如果「指尖」到「手腕」的距離 > 「第二關節(PIP)」到「手腕」的距離，代表伸直
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    extended_fingers = []

    for tip, pip in zip(tips, pips):
        if get_distance(wrist, landmarks[tip]) > get_distance(wrist, landmarks[pip]):
            fingers_up += 1
            extended_fingers.append(tip)

    # 大拇指 (4)：判斷大拇指尖 (4) 距離小指根部 (17) 的距離
    # 如果伸直，大拇指尖會遠離小指根部
    if get_distance(landmarks[4], landmarks[17]) > get_distance(landmarks[3], landmarks[17]):
        fingers_up += 1
        extended_fingers.append(4)

    # 根據伸直的手指來精確判斷 RSP
    if fingers_up == 0:
        return "Rock", (255, 105, 180) # 粉紅色
    elif fingers_up == 5:
        return "Paper", (0, 255, 255) # 黃色
    elif fingers_up == 2 and (8 in extended_fingers and 12 in extended_fingers):
        return "Scissors", (0, 255, 0) # 綠色
    else:
        return "Other", (0, 165, 255) # 橘色


def draw_skeleton(frame, landmarks, w, h):
    """畫出漂亮的彩色骨架"""
    # 定義關節連接線段與顏色
    connections = [
        ([0, 1, 2, 3, 4], (0, 0, 255)),       # 大拇指 (紅)
        ([0, 5, 6, 7, 8], (0, 165, 255)),     # 食指 (橘)
        ([9, 10, 11, 12], (0, 255, 255)),     # 中指 (黃)
        ([13, 14, 15, 16], (0, 255, 0)),      # 無名指 (綠)
        ([0, 17, 18, 19, 20], (255, 0, 0)),   # 小指 (藍)
        ([5, 9, 13, 17], (255, 255, 255))     # 手掌連接 (白)
    ]

    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for connection, color in connections:
        for i in range(len(connection) - 1):
            pt1 = points[connection[i]]
            pt2 = points[connection[i+1]]
            cv2.line(frame, pt1, pt2, color, 3)
            
    for pt in points:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.circle(frame, pt, 3, (0, 0, 0), -1)

# ─────────────────────────────────────────────
# 2. 開啟攝影機進行即時辨識
# ─────────────────────────────────────────────
cap = cv2.VideoCapture(0)
print("📷 攝影機已開啟！按 'q' 鍵退出。")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    label_text = "No Hand"
    color = (0, 0, 255)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            # 畫出骨架
            draw_skeleton(frame, hand_landmarks, w, h)
            
            # 使用強力演算法判斷手勢
            label_text, color = detect_gesture(hand_landmarks)

    # 畫出底色讓字更清楚
    cv2.rectangle(frame, (5, 5), (450, 60), (0, 0, 0), -1)
    cv2.putText(frame, label_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, color, 3, cv2.LINE_AA)

    cv2.imshow("100% MediaPipe - Rock Paper Scissors", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()