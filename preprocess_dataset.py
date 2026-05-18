import os
import cv2
import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─────────────────────────────────────────────
# 設定路徑與超參數
# ─────────────────────────────────────────────
SRC_DIR = 'dataset'
DST_DIR = 'dataset_mediapipe'
IMG_SIZE = 224

# MediaPipe 模型路徑 (由於之前下載在 demo 資料夾)
TASK_PATH = os.path.join('demo', 'hand_landmarker.task')

def letterbox(img, new_shape=(224, 224), color=(0, 0, 0)):
    """保持長寬比縮放圖片，不足的部分填補顏色 (與 carema.py 100% 一致)"""
    shape = img.shape[:2]  # 目前的 [高度, 寬度]
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

def main():
    if not os.path.exists(TASK_PATH):
        print(f"❌ 找不到 MediaPipe 模型 {TASK_PATH}，請確認是否還在。")
        return

    print("⏳ 載入 MediaPipe...")
    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    # 遍歷 train 和 test
    for split in ['train', 'test']:
        split_src = os.path.join(SRC_DIR, split)
        split_dst = os.path.join(DST_DIR, split)

        if not os.path.exists(split_src):
            print(f"⚠️ 找不到資料夾 {split_src}，跳過。")
            continue

        classes = os.listdir(split_src)
        for cls in classes:
            cls_src = os.path.join(split_src, cls)
            if not os.path.isdir(cls_src):
                continue
                
            cls_dst = os.path.join(split_dst, cls)
            os.makedirs(cls_dst, exist_ok=True)

            img_paths = glob.glob(os.path.join(cls_src, '*.jpg')) + glob.glob(os.path.join(cls_src, '*.png'))
            if len(img_paths) == 0:
                continue
                
            print(f"\n📂 開始處理 [{split}/{cls}] (共 {len(img_paths)} 張圖片)...")
            
            success_count = 0
            for i, img_path in enumerate(img_paths):
                # 每 100 張印出進度
                if i > 0 and i % 100 == 0:
                    print(f"  目前進度: {i} / {len(img_paths)}...")

                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                h, w, _ = img.shape
                # MediaPipe 需要 RGB 格式
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
                
                detection_result = detector.detect(mp_image)
                
                # 如果找不到手，就跳過這張圖片
                if not detection_result.hand_landmarks:
                    continue 
                
                # 取第一隻手的特徵點
                hand_landmarks = detection_result.hand_landmarks[0]
                
                # 取得 Bounding Box
                x_min = min([lm.x for lm in hand_landmarks])
                x_max = max([lm.x for lm in hand_landmarks])
                y_min = min([lm.y for lm in hand_landmarks])
                y_max = max([lm.y for lm in hand_landmarks])

                # 與 carema.py 完全一致的 Padding
                small_padding = 15
                px_min = max(0, int(x_min * w) - small_padding)
                px_max = min(w, int(x_max * w) + small_padding)
                py_min = max(0, int(y_min * h) - small_padding)
                py_max = min(h, int(y_max * h) + small_padding)

                if px_max > px_min and py_max > py_min:
                    # 緊湊裁切
                    tight_hand = rgb_img[py_min:py_max, px_min:px_max]
                    
                    # 變成 224x224 正方形 (Letterbox)
                    letterbox_hand = letterbox(tight_hand, new_shape=(IMG_SIZE, IMG_SIZE), color=(0, 0, 0))
                    
                    # 轉回 BGR 以便 cv2 存檔
                    final_bgr = cv2.cvtColor(letterbox_hand, cv2.COLOR_RGB2BGR)
                    
                    filename = os.path.basename(img_path)
                    save_path = os.path.join(cls_dst, filename)
                    cv2.imwrite(save_path, final_bgr)
                    success_count += 1
            
            print(f"✅ 完成 [{split}/{cls}]: 成功處理 {success_count} 張 / 原始 {len(img_paths)} 張 (找不到手而拋棄了 {len(img_paths)-success_count} 張)。")

    print("\n🎉 全部資料集處理完畢！新資料集已儲存至 'dataset_mediapipe' 目錄。")

if __name__ == '__main__':
    main()
