這是一題關於將圖像分割 (Image Segmentation) 模型的輸出應用於實際測量學 (Metrology) 任務的題目。


1. **請用 COCO 資料集 , 挑出有兩隻動物(含或以上)的圖片 , 動物任選**
2. **請用 Segmentation 模型將動物的輪廓 / 動物的眼睛 個別框出**
3. **請量測每隻動物 , 雙眼的距離**
4. **請量測任意兩隻動物 , 右眼的距離**
5. **解釋使用甚麼AI模型 , 為什麼 , 評估標準公式是甚麼:**

### 1. Segmentation: 
YOLOv8x-seg. 負責object detection取得Bounding Box 和 Semantic Segmentation. YOLOv8x-seg可以高度精準的偵測物件和切割輪廓。
### 2. Pose: 
YOLOv8x-pose + AP-10K. AP-10K 標註動物的Keypoints,可以識別動物的眼睛、鼻子等

6. **解釋量測的方式與公式 , 以及怎麼驗證:**

## 距離計算公式 
採用歐幾里得距離（Euclidean Distance）: 針對單一個體雙眼及跨個體右眼分別計算像素距離。 透過`transform_kpt_back_matrix` 轉回原 ROI 座標,因為部分照片需先旋轉ROI（90°/180°/270°）,再偵測keypoints.

### 1. 單一個體雙眼距離 ($D_{\text{intra}}$)

    設單一動物之左眼座標為 `P_L = (X_L, Y_L)`，右眼座標為 `P_R = (X_R, Y_R)`：

* `D_intra = sqrt((X_R - X_L)^2 + (Y_R - Y_L)^2)`

### 2. 跨個體右眼距離 ($D_{\text{inter}}$)

    對畫面中所有偵測到右眼的動物按 X 軸座標遞增排序。設前兩隻動物之右眼座標分別為 `P_R1 = (X_R1, Y_R1)` 與 `P_R2 = (X_R2, Y_R2)`：

* `D_inter = sqrt((X_R2 - X_R1)^2 + (Y_R2 - Y_R1)^2)`


## 視覺化輸出說明

* **綠色外框 (Green Outline)**：分割模型繪製之動物區域輪廓。
* **黃線與點 (Yellow Line & Points)**：單一個體的雙眼連線與關鍵點，標示文字為 `A{index}: {D_intra}px`。
* **紅線**：相鄰兩隻動物右眼間的連線，標示文字為 `Inter R-Eye: {D_inter}px`。


## Model Weights

執行本程式前，請確保以下模型權重檔已下載並放置於 `.env` 所設定的 `MODEL_DIR` 目錄中：

1. **YOLOv8 Segmentation**：`yolov8x-seg.pt`
2. **YOLOv8 Pose **：`yolov8x-pose.pt`
3. **AP-10K (MMPose)**：
   * 請至 [MMPose AP-10K Model Zoo](https://github.com/open-mmlab/mmpose/blob/main/configs/animal_2d_keypoint/rtmpose/ap10k/rtmpose_ap10k.md) 下載對應的檔案。
   * 下載後請將檔案重命名為 `ap10k_pt.pth` 並放入模型資料夾。

## 運行步驟
` python metrology.py`