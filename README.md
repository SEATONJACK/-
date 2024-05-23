# 圖形識別實務與應用作業

> [!NOTE] 
> 1. 需要注意，這裡面隱藏層只有用一層的神經元
> 2. 程式碼的簡潔程度依據 期中->期末 的順序變得越來越簡潔，所以想看護眼睛的，可以往後看

## 期中作業 (iris集合的分類問題)

* 主要使用和學習類神經網路和倒傳遞
* `midterm.py` 是類神經網路的初次實作
* `midterm_1.py` 主要是將`midterm.py`的實作再加上one hot encoding，以增加class的識別率
*  `homework_to_exam.py` 主要將softmax以及cross entropy的概念加進來，用來預習作業(依舊使用iris集合)

## 期末作業 (灰階圖片(.bmp)的分類問題)

* `final_homework_dl.py` 主要是將灰階圖以pca降維、以min max歸一化後，再以先前的類神經網路來進行訓練和判別。
* `final_homework_lda.py` 主要是將灰階圖片以pca降維後，再以lda將其中各個class的between提升，最後以nearest neighbors判斷測試資料屬於的資料集
