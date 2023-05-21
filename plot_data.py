import matplotlib.pyplot as plt
import numpy as np


def getdata(data_loc):
    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    acc_list = []
    with open(data_loc, "r") as f:
        for i in f.readlines():
            data_i = i.split("\t")
            epoch_i = float(data_i[0][7:])
            train_loss_i = float(data_i[1][10:])
            test_loss_i = float(data_i[2][9:])
            acc_i = float(data_i[3][13:])
            epoch_list.append(epoch_i)
            train_loss_list.append(train_loss_i)
            test_loss_list.append(test_loss_i)
            acc_list.append(acc_i)
        print(len(epoch_list), len(train_loss_list))
        return epoch_list, train_loss_list, test_loss_list, acc_list



if __name__ == "__main__":
    data_loc = r"mobilenet_36_traindata.txt"
    epoch_list, train_loss_list, test_loss_list, acc_list = getdata(data_loc)
    #训练损失显示
    plt.plot(epoch_list, train_loss_list)
    plt.legend(["model"])
    plt.xticks(np.arange(0, 50, 5))  # 横坐标的值和步长
    plt.yticks(np.arange(0, 100, 10))  # 横坐标的值和步长
    plt.xlabel("Epoch")
    plt.ylabel("train_loss")
    plt.title("Model Loss")
    plt.show()

    #训练准确率变化
    plt.plot(epoch_list, acc_list)
    plt.legend(["model"])
    plt.xticks(np.arange(0, 50, 5))  # 横坐标的值和步长
    plt.yticks(np.arange(0, 100, 10))  # 横坐标的值和步长
    plt.xlabel("Epoch")
    plt.ylabel("acc_list")
    plt.title("Model Accuracy")
    plt.show()
