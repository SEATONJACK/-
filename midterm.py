import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def logsig(x):
    return 1/(1+np.exp(-x))

def dlogsig(x):
    return x*(1-x)

def RMSE(inp, out):
    return np.sqrt()


def perceptron():
    with open("iris_in.csv", 'r') as Fin, open("iris_out.csv", 'r') as Fout:
        inp_csv = np.loadtxt(Fin, delimiter=',')
        epoch = 15
        neuronNumber = 12

        t_list = [1, 2, 3]
        inp_train = inp_csv[:75, :]
        inp_test = inp_csv[75:, :]

        Whid = np.random.normal(0, 0.5, size=(inp_train.shape[1], neuronNumber))
        Wout = np.random.normal(0, 0.5, size=(neuronNumber, 1))

        # 訓練模型
        RMSE_list = []
        for round in range(0, epoch):
            MSE = 0
            for i in range(0, 75):
                t = t_list[i%3]
                inp = inp_train[i, :]

                SUMhid = inp @ Whid
                Ahid = logsig(SUMhid)

                SUMout = Ahid @ Wout
                Aout = SUMout

                MSE += (1 - Aout[0]) ** 2

                DELTAout = (t - Aout[0]) * 1    # 這裡的1指得是dpuline()的結果
                DELTAhid = DELTAout*Wout

                Wout = Wout + (DELTAout*Ahid).reshape(neuronNumber, 1)

                DELTA_dlogsig = (DELTAhid*(dlogsig(Ahid).reshape(Ahid.shape[0], 1))).reshape(1, DELTAhid.shape[0])
                DELTA_dlogsig = np.repeat(DELTA_dlogsig[0:1], inp.shape[0], axis=0)
                inp = inp.reshape(-1, 1)
                inp = np.repeat(inp[:, 0:1], neuronNumber, axis=1)

                Whid = Whid + DELTA_dlogsig*inp

            RMSE_list.append(MSE**(1/2)/75)
            print(f"Round {round} RMSE: {MSE**(1/2)/75:.{3}f}")

        # 驗證模型
        counter = 0
        for i in range(0, 75):
            t = t_list[i%3]
            test = inp_test[i, :]

            SUMhid = np.dot(test, Whid)
            Ahid = logsig(SUMhid)

            SUMout = np.dot(Ahid, Wout)
            Aout = SUMout

            if abs(Aout[0]-t) <= 0.5:
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