import numpy as np
from PIL import Image
import os
from scipy.spatial.distance import cdist


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

class LDA:
    def __init__(self, train_set, train_set_dimension, classNum, ldaDimension):
        self.train_set = train_set
        self.train_set_dimension = train_set_dimension
        self.total_data_num = train_set.shape[0]
        self.dataNum_each_Class = int(train_set.shape[1]/classNum)
        self.ldaDimension = ldaDimension

    def fit_transform(self):
        train_mean_set = []
        train_overall_mean = np.mean(self.train_set, axis=0)
        # 求各個class的平均
        for ind in range(0, self.total_data_num, self.dataNum_each_Class):
            train_mean_set.append(np.mean(self.train_set[ind:ind + 5, :], axis=0))

        # 計算sb矩陣
        n_features = self.train_set_dimension  # 這裡是features的數量
        Sb = np.zeros((n_features, n_features))
        for ind in range(0, int(self.total_data_num/self.dataNum_each_Class)):
            n_i = self.dataNum_each_Class
            n = train_mean_set[ind] - train_overall_mean
            Sb += n_i * np.outer(n, n)

        evals, evecs = np.linalg.eigh(Sb)
        evals = np.real(evals)
        evecs = np.real(evecs)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]

        lda_projections = evecs[:, :self.ldaDimension]

        return lda_projections


def run():
    folder_num = 40
    dimension = 200     # 此處也是降維後的維度
    lda_dimension = 65
    dataNum_each_Class = 5
    total_data_num = folder_num * dataNum_each_Class

    pca = PCA(n_components=dimension)

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
    train_set = pca.fit_transform(np.vstack(train_set))
    test_set = pca.fit_transform(np.vstack(test_set))

    lda = LDA(train_set, dimension, folder_num, lda_dimension)
    lda_projection = lda.fit_transform()
    train_set = train_set @ lda_projection
    test_set = test_set @ lda_projection

    train_mean_set = []
    for i in range(0, total_data_num, 5):
        train_mean_set.append(np.mean(train_set[i:i+5, :], axis=0))

    counter = 0
    for i in range(total_data_num):
        # 計算每個向量與q的歐式距離
        distances = cdist(train_set, [test_set[i]], 'euclidean')

        # 找出最近的row index
        nearest_idx = np.argmin(distances)

        if int(nearest_idx/5) == int(i/5):
            counter += 1

    # print(counter)
    success_rate = counter / total_data_num
    print(f"符合率: {success_rate * 100:.{2}f}%")


if __name__ == "__main__":
    run()