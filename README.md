## 刷機

https://www.raspberrypi.com/software/

1.	下載 Raspberry Pi Imager
    ![alt text](image.png)
2.	選擇 Raspberry Pi 4 64-bit
    ![alt text](image-1.png)
3.	插上讀卡機
    ![alt text](image-2.png)
4.	輸入主機名，ssh 會用到
    ![alt text](image-3.png)
5.	首都 Taipei ;時區Asia/Taipei
    ![alt text](image-4.png)
6.	輸入用戶名及密碼，ssh 會用到
    ![alt text](image-5.png)
7.	開啟 ssh
    ![alt text](image-6.png)
8.	完成寫入
    ![alt text](image-7.png)

---

## 運行

https://github.com/BiBaIsAFish/RSP_demo

1.	下載 github
2.	Demo carema
    ![](17903.jpg)
    ![](17904.jpg)

---

## 評分標準

- 成功在 Raspberry Pi 4 上執行 test.py & carema.py 50%
- Demo 展示影片(carema) 15%
    - 從兩個模型中選擇較強的模型，寫一支程式將 carema 接收到的畫面接到模型上進行分類，並錄一段 demo 10 個手勢的短片
    - 執行 10 次手勢辨識，且須包含以下手勢
        - 石頭(Rock)
        - 剪刀(Scissors)
        - 布(Paper)
        - 其他錯誤手勢 (Error)
- 報告 35%
	- 需自行找兩個模型架構修改 20%
		- 至少需呈現 accuracy, precision, recall, F1-score
	- 需解釋更換模型原因及比較差異 15%
