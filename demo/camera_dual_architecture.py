"""
camera_dual_architecture.py
===========================
雙重架構版本：
1. 先用 MediaPipe 偵測手部，並取得裁切邊界。
2. 將長方形的裁切影像透過 Letterbox (補黑邊) 變成 224x224 的完美正方形。
3. 餵給我們自己訓練好的 EfficientNet-B0 進行神經網路辨識。
"""

import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ─────────────────────────────────────────────
# 1. 初始化 EfficientNet-B0 模型
# ─────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'rps_efficientnet.pth'

print("⏳ 載入 EfficientNet-B0 模型中...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ 找不到模型檔案 {MODEL_PATH}！")
    exit()

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
CLASSES = checkpoint['classes']
NUM_CLASSES = checkpoint['num_classes']
IMG_SIZE = checkpoint['img_size']

model = models.efficientnet_b0(weights=None)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(256, NUM_CLASSES),
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

print("✅ 模型載入完成！")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# 2. 初始化 MediaPipe (Tasks API)
# ─────────────────────────────────────────────
print("⏳ 初始化 MediaPipe...")
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

def letterbox(img, new_shape=(224, 224), color=(0, 0, 0)):
    """保持長寬比縮放圖片，不足的部分填補顏色 (Letterbox)"""
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img

def draw_skeleton(frame, landmarks, w, h):
    connections = [
        ([0, 1, 2, 3, 4], (0, 0, 255)),
        ([0, 5, 6, 7, 8], (0, 165, 255)),
        ([9, 10, 11, 12], (0, 255, 255)),
        ([13, 14, 15, 16], (0, 255, 0)),
        ([0, 17, 18, 19, 20], (255, 0, 0)),
        ([5, 9, 13, 17], (255, 255, 255))
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
# 3. 開啟攝影機進行即時辨識
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
            draw_skeleton(frame, hand_landmarks, w, h)
            
            # 取得手部 Bounding Box
            x_min = min([lm.x for lm in hand_landmarks])
            x_max = max([lm.x for lm in hand_landmarks])
            y_min = min([lm.y for lm in hand_landmarks])
            y_max = max([lm.y for lm in hand_landmarks])

            small_padding = 15
            px_min = max(0, int(x_min * w) - small_padding)
            px_max = min(w, int(x_max * w) + small_padding)
            py_min = max(0, int(y_min * h) - small_padding)
            py_max = min(h, int(y_max * h) + small_padding)

            if px_max > px_min and py_max > py_min:
                tight_hand = rgb_frame[py_min:py_max, px_min:px_max]
                letterbox_hand = letterbox(tight_hand, new_shape=(IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
                
                pil_img = Image.fromarray(letterbox_hand)
                input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, pred = torch.max(probs, 1)
                    pred_class = CLASSES[pred.item()]
                    conf_val = conf.item() * 100

                if pred_class.lower() == 'rock':
                    label_text = f"Rock ({conf_val:.1f}%)"
                    color = (255, 105, 180)
                elif pred_class.lower() == 'paper':
                    label_text = f"Paper ({conf_val:.1f}%)"
                    color = (0, 255, 255)
                elif pred_class.lower() == 'scissors':
                    label_text = f"Scissors ({conf_val:.1f}%)"
                    color = (0, 255, 0)
                else:
                    label_text = f"Other ({conf_val:.1f}%)"
                    color = (0, 0, 255)

                cv2.rectangle(frame, (px_min, py_min), (px_max, py_max), (255, 255, 255), 2)

    cv2.rectangle(frame, (5, 5), (450, 60), (0, 0, 0), -1)
    cv2.putText(frame, label_text, (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    cv2.imshow("Dual Architecture - EfficientNet", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
