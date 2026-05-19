"""
extract_landmarks.py
--------------------
從 dataset/ 資料夾內的圖片，用 MediaPipe 抽取 21 個手部關節點 (landmark)，
將座標儲存成 CSV 供後續訓練使用。

輸入結構:
    dataset/
        train/
            rock/  scissors/  paper/
        test/
            rock/  scissors/  paper/

輸出:
    landmark_data/
        train.csv
        test.csv

每列格式:
    label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20
    (共 1 + 21*3 = 64 欄)
"""

import os
import cv2
import glob
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── 設定 ──────────────────────────────────────
SRC_DIR   = 'dataset'
DST_DIR   = 'landmark_data'
TASK_PATH = os.path.join('demo', 'hand_landmarker.task')

CLASSES = ['rock', 'scissors', 'paper']   # 依你的資料夾名稱調整
# ─────────────────────────────────────────────


def normalize_landmarks(hand_landmarks):
    """
    對 21 個關節點做正規化，讓模型不受手的絕對位置與大小影響。

    步驟:
    1. 以手腕 (landmark 0) 為原點，平移所有點
    2. 以手腕到中指根部 (landmark 9) 的距離做縮放
    3. 回傳長度為 63 的 flat list [x0,y0,z0, x1,y1,z1, ...]
    """
    wrist = hand_landmarks[0]
    ref_x, ref_y, ref_z = wrist.x, wrist.y, wrist.z

    # 計算參考距離 (手腕 → 中指根部)
    mid_mcp = hand_landmarks[9]
    scale = ((mid_mcp.x - ref_x)**2 +
             (mid_mcp.y - ref_y)**2 +
             (mid_mcp.z - ref_z)**2) ** 0.5 + 1e-6

    coords = []
    for lm in hand_landmarks:
        coords.append((lm.x - ref_x) / scale)
        coords.append((lm.y - ref_y) / scale)
        coords.append((lm.z - ref_z) / scale)
    return coords


def process_split(detector, split):
    split_src = os.path.join(SRC_DIR, split)
    if not os.path.exists(split_src):
        print(f"⚠️  找不到 {split_src}，跳過。")
        return

    os.makedirs(DST_DIR, exist_ok=True)
    out_csv = os.path.join(DST_DIR, f'{split}.csv')

    # 欄位名稱
    header = ['label']
    for i in range(21):
        header += [f'x{i}', f'y{i}', f'z{i}']

    total_ok = 0
    total_skip = 0

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for cls in CLASSES:
            cls_path = os.path.join(split_src, cls)
            if not os.path.isdir(cls_path):
                print(f"⚠️  找不到類別資料夾 {cls_path}，跳過。")
                continue

            img_paths = (glob.glob(os.path.join(cls_path, '*.jpg')) +
                         glob.glob(os.path.join(cls_path, '*.jpeg')) +
                         glob.glob(os.path.join(cls_path, '*.png')))

            print(f"\n📂 [{split}/{cls}]  共 {len(img_paths)} 張...")

            ok = skip = 0
            for i, img_path in enumerate(img_paths):
                if i % 200 == 0 and i > 0:
                    print(f"   進度 {i}/{len(img_paths)}")

                img = cv2.imread(img_path)
                if img is None:
                    skip += 1
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = detector.detect(mp_img)

                if not result.hand_landmarks:
                    skip += 1
                    continue

                coords = normalize_landmarks(result.hand_landmarks[0])
                writer.writerow([cls] + coords)
                ok += 1

            print(f"   ✅ 成功 {ok}  跳過(無手) {skip}")
            total_ok += ok
            total_skip += skip

    print(f"\n💾 [{split}] 完成 → {out_csv}")
    print(f"   總計: 成功 {total_ok}  跳過 {total_skip}")


def main():
    if not os.path.exists(TASK_PATH):
        print(f"❌ 找不到 MediaPipe 模型: {TASK_PATH}")
        return

    print("⏳ 載入 MediaPipe Hand Landmarker...")
    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    for split in ['train', 'test']:
        process_split(detector, split)

    print("\n🎉 全部完成！資料儲存在 landmark_data/ 目錄。")


if __name__ == '__main__':
    main()
