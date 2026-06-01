import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import cv2
import glob
import csv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'dataset')
TASK_PATH = os.path.join(BASE_DIR, 'demo', 'hand_landmarker.task')

def normalize_landmarks(landmarks):
    """
    將 MediaPipe 的 21 個座標點進行「平移」、「旋轉」與「縮放」歸一化，
    確保不管手在畫面的哪裡、轉向什麼角度、離鏡頭多遠，座標特徵都完全一致。
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # 1. 平移歸一化：以手腕 (第0點) 為原點 (0, 0, 0)
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
    coords = coords @ R.T  # 將所有點繞 Z 軸旋轉
    
    # 3. 縮放歸一化：找出距離手腕最遠的點的距離
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords = coords / max_dist
        
    return coords.flatten().tolist()

def main():
    if not os.path.exists(TASK_PATH):
        print(f"❌ 找不到 MediaPipe 模型 {TASK_PATH}")
        return

    print("⏳ 初始化 MediaPipe...")
    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    for split in ['train', 'test']:
        split_src = os.path.join(SRC_DIR, split)
        if not os.path.exists(split_src):
            continue

        data = []
        classes = ['rock', 'paper', 'scissors']
        
        for label_idx, cls in enumerate(classes):
            cls_src = os.path.join(split_src, cls)
            if not os.path.isdir(cls_src):
                continue
                
            img_paths = glob.glob(os.path.join(cls_src, '*.jpg')) + glob.glob(os.path.join(cls_src, '*.png'))
            print(f"📂 正在轉換 [{split}/{cls}] ({len(img_paths)} 張圖片)...")
            
            for img_path in img_paths:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                detection_result = detector.detect(mp_image)
                
                # 如果有抓到手，就轉換座標並存起來
                if detection_result.hand_landmarks:
                    features = normalize_landmarks(detection_result.hand_landmarks[0])
                    data.append(features + [label_idx])
                    
        # 將抓出的特徵存成 CSV
        columns = [f"f_{i}" for i in range(63)] + ["label"]
        csv_path = os.path.join(BASE_DIR, 'train', f"{split}_landmarks.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(data)
            
        print(f"✅ 成功儲存 {csv_path} (共 {len(data)} 筆座標資料)\n")

if __name__ == '__main__':
    main()
