import os
import cv2
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. 設定模型與資料集路徑
    model_path = 'rps_svm_model.pkl'
    # 假設 demo 資料夾與 dataset 資料夾在同一個主目錄下
    test_dir = '../dataset/test' 

    if not os.path.exists(model_path):
        print(f"❌ 錯誤：找不到模型檔案 '{model_path}'，請確認是否已放入 demo 資料夾。")
        return
    
    if not os.path.exists(test_dir):
        print(f"❌ 錯誤：找不到測試資料集 '{test_dir}'。")
        print("請確認 dataset/test 資料夾是否存在於上一層目錄。")
        return

    # 2. 載入模型
    print("⏳ 載入模型中...")
    clf = joblib.load(model_path)
    print("✅ 模型載入成功！\n")

    label_map = {'rock': 0, 'paper': 1, 'scissors': 2}
    X_test, y_test = [], []

    # 3. 讀取並處理測試圖片
    print("📂 正在讀取測試集圖片並進行預測...")
    for category, label_idx in label_map.items():
        category_path = os.path.join(test_dir, category)
        
        # 處理資料夾內可能多包一層的情況
        if not os.path.exists(category_path):
            subdirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
            if subdirs:
                category_path = os.path.join(test_dir, subdirs[0], category)

        if not os.path.exists(category_path):
            print(f"⚠️ 找不到 {category} 的圖片資料夾，略過...")
            continue
            
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(category_path, filename)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # 資料前處理 (必須與訓練時一致)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (64, 64))
                    X_test.append(resized.flatten())
                    y_test.append(label_idx)

    if not X_test:
        print("❌ 錯誤：沒有讀取到任何圖片，請檢查資料夾結構。")
        return

    # 正規化
    X_test = np.array(X_test) / 255.0
    y_test = np.array(y_test)

    # 4. 進行預測與評估
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 測試結果統整:")
    print(f"總共測試了 {len(y_test)} 張圖片")
    print(f"🎯 模型準確率: {accuracy * 100:.2f}%\n")
    print("📝 分類詳細報告:")
    print(classification_report(y_test, y_pred, target_names=['Rock', 'Paper', 'Scissors']))

if __name__ == "__main__":
    main()