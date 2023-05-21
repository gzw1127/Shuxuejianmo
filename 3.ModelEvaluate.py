'''
    1.单幅图片验证
    2.多幅图片验证
'''
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet34
from utils import LoadData, write_result
import pandas as pd


def eval(dataloader, model):
    label_list = []
    likelihood_list = []
    model.eval()
    with torch.no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            # 将数据转到GPU
            X = X.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # 获取可能性最大的标签
            label = torch.softmax(pred,1).cpu().numpy().argmax()
            label_list.append(label)
            # 获取可能性最大的值（即概率）
            likelihood = torch.softmax(pred,1).cpu().numpy().max()
            likelihood_list.append(likelihood)
        return label_list,likelihood_list


if __name__ == "__main__":

    '''
    1. 导入模型结构
    '''
    model = resnet34(pretrained=False)
    num_ftrs = model.fc.in_features    # 获取全连接层的输入
    model.fc = nn.Linear(num_ftrs, 5)  # 全连接层改为不同的输出
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    '''
    2. 加载模型参数
    '''
    model_loc = "./BEST_resnet_epoch_41_acc_85.9.pth"
    model_dict = torch.load(model_loc)
    model.load_state_dict(model_dict)
    model = model.to(device)

    '''
    3. 加载图片
    '''
    #eval相当于测试集
    valid_data = LoadData("eval.txt", train_flag=False)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=2, pin_memory=True, batch_size=1)


    '''
    4. 获取结果
    '''
    label_list, likelihood_list =  eval(test_dataloader, model)
    label_names = ["daisy", "dandelion","rose","sunflower","tulip"]

    result_names = [label_names[i] for i in label_list]

    list = [result_names, likelihood_list]
    df = pd.DataFrame(data=list)
    df2 = pd.DataFrame(df.values.T, columns=["label", "likelihood"])
    print(df2)
    df2.to_csv('testdata.csv', encoding='gbk')




