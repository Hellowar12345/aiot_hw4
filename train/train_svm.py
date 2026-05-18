import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def load_images_from_folder(folder_path):
    """讀取本地資料夾內的圖片並轉換為 SVM 可用的陣列"""
    images = []
    labels = []
    # 定義標籤對應 (需與 demo 端的標籤順序一致: 0=Rock, 1=Paper, 2=Scissors)
    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    
    for category, label_idx in label_map.items():
        # 組合路徑，例如 dataset/train/rock
        category_path = os.path.join(folder_path, category)
        
        # 處理官方壓縮檔解壓縮後可能多包一層資料夾(如 rps/) 的情況
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
            if subdirs:
                category_path = os.path.join(folder_path, subdirs[0], category)
        
        if not os.path.exists(category_path):
            print(f"⚠️ 警告: 找不到 {category} 的資料夾 -> {category_path}")
            continue
            
        print(f"📂 正在載入 {category} 的圖片...")
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # 1. 轉灰階
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # 2. 縮放至 64x64
                    resized = cv2.resize(gray, (64, 64))
                    # 3. 攤平為一維陣列
                    images.append(resized.flatten())
                    labels.append(label_idx)
                    
    # 將像素值除以 255.0 進行正規化
    return np.array(images) / 255.0, np.array(labels)

def main():
    # 動態取得專案根目錄 (也就是 train_svm.py 所在資料夾的上一層)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 定義資料夾路徑
    train_dir = os.path.join(base_dir, 'dataset', 'train')
    test_dir = os.path.join(base_dir, 'dataset', 'test')
    demo_dir = os.path.join(base_dir, 'demo')

    print("=== 步驟 1: 開始讀取本機圖片 ===")
    X_train, y_train = load_images_from_folder(train_dir)
    print("---")
    X_test, y_test = load_images_from_folder(test_dir)

    print(f"\n📊 讀取完成！訓練樣本數: {len(X_train)}, 測試樣本數: {len(X_test)}")

    if len(X_train) == 0:
        print("❌ 錯誤：找不到任何圖片，請檢查 dataset 資料夾的結構是否正確！")
        return

    print("\n=== 步驟 2: 開始訓練 SVM 模型 (這可能需要一到兩分鐘) ===")
    clf = SVC(kernel='rbf', C=1.0, gamma='scale')
    clf.fit(X_train, y_train)

    print("\n=== 步驟 3: 評估模型 ===")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"🎯 模型準確率: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

    # 確保 demo 資料夾存在
    os.makedirs(demo_dir, exist_ok=True)
    
    # 將模型直接存進 demo 資料夾
    model_path = os.path.join(demo_dir, 'rps_svm_model.pkl')
    joblib.dump(clf, model_path)
    print(f"✅ 模型已成功儲存並自動放置於: {model_path}")

if __name__ == "__main__":
    main()