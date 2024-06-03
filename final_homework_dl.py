import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def logsig(x):
    return 1/(1+np.exp(-x))

def dlogsig(x):
    return x*(1-x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.min_vals = None
        self.max_vals = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.components = None

    def pre_calculation(self, X):
        # 均值歸一化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 計算協方差矩陣
        cov = np.cov(X_centered.T)

        # 對協方差矩陣進行特徵值分解
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        self.eigenvalues = np.real(eigenvalues)
        self.eigenvectors = np.real(eigenvectors)

        # 排序特徵值並選取前n_components個主成分
        idx = np.argsort(self.eigenvalues)[::-1]
        idx = idx[:self.n_components]
        self.components = self.eigenvectors[:, idx]


    def fit_transform(self, X):
        # 均值歸一化
        X_centered = X - self.mean

        # 降維
        x_resize = np.dot(X_centered, self.components)

        return x_resize

    def find_maxmin(self, X):
        self.min_vals = np.min(X, axis=0)
        self.max_vals = np.max(X, axis=0)

    def min_max_normalize(self, X):
        # 進行Min-Max標準化
        X_normalized = (X - self.min_vals) / (self.max_vals - self.min_vals)

        return X_normalized


def perceptron():
    folder_num = 40
    epoch = 150
    learning_rate = 0.1
    dataNum_each_Class = 5

    total_data_num = folder_num * dataNum_each_Class
    neuronNumber_inp = 65  # 此處也是降維後的維度
    neuronNumber_hid = 150
    neuronNumber_out = folder_num

    pca = PCA(n_components=neuronNumber_inp)

    train_set = []
    test_set = []

    # 讀取檔案
    # 這裡要注意的是，因為讀檔案時，會把開頭的數字看成字串
    # 所以是以dictionary的方式來排序的（也就是1 -> 10 -> 11...）
    # 但因為這裡每個類別並不需要對其組別有嚴格的對應，所以這裡就不做修改了
    # （還有一個原因是，訓練組和對照組都是在同一個folder內，所以沒差）
    for root, dirs, files in os.walk("ORL3232"):
        for f in files:
            # 將奇怪的檔案過濾掉
            if f.split('.')[1] != "bmp":
                continue

            img_ind = int(f.split('.')[0])

            full_path = os.path.join(root, f)
            with Image.open(full_path) as img:
                img_array = np.array(img)

                # 將二維陣列攤平為一維陣列
                flat_arr = img_array.reshape(1, -1)  # -1 表示自動計算行數

                if img_ind % 2 == 1:
                    train_set.append(flat_arr)
                else:
                    test_set.append(flat_arr)

    # 降維
    pca.pre_calculation(np.vstack(train_set))
    train_PCAed = pca.fit_transform(np.vstack(train_set))
    pca.find_maxmin(train_PCAed)
    train_set = pca.min_max_normalize(train_PCAed)

    test_PCAed = pca.fit_transform(np.vstack(test_set))
    test_set = pca.min_max_normalize(test_PCAed)

    # 權重矩陣
    Whid = np.random.normal(0, 0.1, size=(neuronNumber_inp, neuronNumber_hid))
    Wout = np.random.normal(0, 0.1, size=(neuronNumber_hid, neuronNumber_out))

    # 訓練模型
    cross_entropy_list = []
    for round in range(epoch):
        cross_entropy = 0

        for i in range(total_data_num):
                inp = train_set[i, :]

                SUMhid = inp @ Whid
                Ahid = logsig(SUMhid)

                SUMout = Ahid @ Wout
                Aout = softmax(SUMout)

                # 計算cross entropy
                cross_entropy += -np.log(Aout[int(i/5)])

                # 以下是倒傳遞的過程
                DELTAout = Aout.copy()
                DELTAout[int(i / 5)] -= 1
                # 這裡要注意，原先因為是負梯度，所以直接加上去就可以了，這次的值是正的，所以要加負號
                DELTAout = -1 * DELTAout
                DELTAhid = DELTAout @ Wout.T * dlogsig(Ahid)

                '''
                因為我在這裡卡了蠻多天的，所以我覺得他需要特別一個區塊來講我有多麼痛苦
                這裡需要 特別! 特別! 特別! 萬分! 萬分! 萬分! 注意! 注意! 注意!
                因為整體的更新資料實在太大了，過於激進的資料可能使得整體的更新過大，進而使得權重矩陣出錯
                所以需要將我們Delta的結果乘上learning_rate（可以自己訂）以避免發生以上的錯誤
                (要再注意epoch過多，導致過度擬合的狀況)
                '''
                Wout += np.outer(Ahid, DELTAout * learning_rate)
                Whid += np.outer(inp, DELTAhid * learning_rate)

        cross_entropy_list.append(cross_entropy/total_data_num)
        print(f"Round {round+1} cross_entropy: {cross_entropy/total_data_num:.{3}f}")

    # 驗證模型
    counter = 0
    for i in range(total_data_num):
        inp = test_set[i, :]

        SUMhid = inp @ Whid
        Ahid = logsig(SUMhid)

        SUMout = Ahid @ Wout
        Aout = softmax(SUMout)
        print(Aout)

        if np.argmax(Aout) == int(i/5):
            # print(f"實際輸出：{Aout}    應該輸出：{t_matrix[i%3]}")
            counter += 1

    success_rate = counter/total_data_num
    print(f"符合率: {success_rate*100:.{2}f}%")

    # 繪製折線圖
    plt.plot(range(len(cross_entropy_list)), cross_entropy_list, marker='.', linestyle='-')

    # 添加標題和標籤
    plt.title('Line Chart')
    plt.xlabel('epoch')
    plt.ylabel('cross entropy')

    # 顯示網格
    plt.grid(True)

    # 顯示圖表
    plt.show()

if __name__ == "__main__":
    perceptron()