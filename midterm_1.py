import numpy as np
import matplotlib.pyplot as plt

def logsig(x):
    return 1/(1+np.exp(-x))

def dlogsig(x):
    return x*(1-x)

def RMSE(inp, out):
    return np.sqrt()

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

def perceptron():
    with open("iris_in.csv", 'r') as Fin, open("iris_out.csv", 'r') as Fout:
        inp_csv = np.loadtxt(Fin, delimiter=',')
        epoch = 18
        neuronNumber_hid = 20
        neuronNumber_out = 3

        t_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        inp_train = inp_csv[:75, :]
        inp_test = inp_csv[75:, :]

        Whid = np.random.normal(0, 0.5, size=(inp_train.shape[1], neuronNumber_hid))
        Wout = np.random.normal(0, 0.5, size=(neuronNumber_hid, neuronNumber_out))

        # 訓練模型
        RMSE_list = []
        for round in range(0, epoch):
            MSE = [0, 0, 0]
            for i in range(0, 75):
                t = t_matrix[i%3: i%3+1]
                inp = inp_train[i, :]

                SUMhid = inp @ Whid
                Ahid = logsig(SUMhid)

                SUMout = Ahid @ Wout
                Aout = softmax(SUMout)

                MSE += (Aout - t) ** 2

                DELTAout = Aout.copy()
                DELTAout[i % 3] -= 1
                # 這裡要注意，因為之前是負梯度，所以直接加上去就可以了，這次的值是正的，所以要加負號
                DELTAout = -1 * DELTAout
                DELTAhid = DELTAout @ Wout.T * dlogsig(Ahid)

                # 這裡要注意，因為之前是負梯度，所以直接加上去就可以了，這次的值是正的，所以要用減的
                Wout += np.outer(Ahid, DELTAout)
                Whid += np.outer(inp, DELTAhid)


            print(MSE[0, 2])
            RMSE_list.append((MSE[0, 0]**(1/2) + MSE[0, 1]**(1/2) + MSE[0, 2]**(1/2))/(3*75))
            print(f"Round {round+1} RMSE: {(MSE[0, 0]**(1/2) + MSE[0, 1]**(1/2) + MSE[0, 2]**(1/2))/225:.{3}f}")

        # 驗證模型
        counter = 0
        for i in range(0, 75):
            t = t_matrix[i % 3: i % 3 + 1]
            inp = inp_train[i, :]

            SUMhid = np.dot(inp, Whid)
            Ahid = logsig(SUMhid)

            SUMout = np.dot(Ahid, Wout)
            Aout = logsig(SUMout)

            if np.argmax(Aout) == i%3:
                print(f"實際輸出：{Aout}    應該數出：{t_matrix[i%3]}")
                counter += 1

        success_rate = counter/75

        print(f"符合率: {success_rate*100:.{2}f}%")

        # 繪製折線圖
        plt.plot(range(len(RMSE_list)), RMSE_list, marker='.', linestyle='-')

        # 添加標題和標籤
        plt.title('Line Chart')
        plt.xlabel('epoch')
        plt.ylabel('RMSE')

        # 顯示網格
        plt.grid(True)

        # 顯示圖表
        plt.show()

if __name__ == "__main__":
    perceptron()