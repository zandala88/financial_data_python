import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from inf.utils.timefeatures import time_features
from models import Informer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    """
    数据加载器函数，用于加载和预处理时间序列数据，以用于训练模型。

    仅仅适用于 多变量预测多变量（可以单独取单变量的输出），或者单变量预测单变量
    也就是y里也会有外生变量？？

    参数:
    - window: 窗口大小，用于截取输入序列的长度。
    - length_size: 目标序列的长度。
    - batch_size: 批量大小，决定每个训练批次包含的数据样本数量。
    - data: 输入时间序列数据。
    - data_mark: 输入时间序列的数据标记，用于辅助模型训练或增加模型的多样性。

    返回值:
    - dataloader: 数据加载器，用于批量加载处理后的训练数据。
    - x_temp: 处理后的输入数据。
    - y_temp: 处理后的目标数据。
    - x_temp_mark: 处理后的输入数据的标记。
    - y_temp_mark: 处理后的目标数据的标记。
    """

    # 构建模型的输入
    seq_len = window
    sequence_length = seq_len
    result = np.array([data[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])
    result_mark = np.array([data_mark[i: i + sequence_length] for i in range(len(data) - sequence_length + 1)])

    # 划分x与y
    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]

    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    # 转换为Tensor和数据类型
    x_temp = torch.tensor(x_temp).type(torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32)
    y_temp = torch.tensor(y_temp).type(torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark

window = 10  # 模型输入序列长度
length_size = 1  # 预测结果的序列长度
batch_size = 32



df = pd.read_csv('data\\test_data_more.csv') # 将你所需要的数据放到data文件夹，然后只要改成你自己的文件名
# 注意多变量情况下，目标变量必须为最后一列
data_dim = df[df.columns.drop('date')].shape[1]  # 一共多少个变量 这个不去动
data_target = df['Target']  # 预测的目标变量 把预测的列名改为target
data = df[df.columns.drop('date')]  # 选取所有的数据

df_stamp = df[['date']]
df_stamp['date'] = pd.to_datetime(df_stamp.date)
data_stamp = time_features(df_stamp, timeenc=1, freq='B')  # 这一步很关键，注意数据的freq 你的数据是最小精度是什么就填什么，下面有

scaler = MinMaxScaler()
data_inverse = scaler.fit_transform(np.array(data))

data_length = len(data_inverse)
train_set = 0.7

data_train = data_inverse[:int(train_set * data_length), :]  # 读取目标数据，第一列记为0：1，后面以此类推, 训练集和验证集，如果是多维输入的话最后一列为目标列数据
data_train_mark = data_stamp[:int(train_set * data_length), :]
data_test = data_inverse[:, :]  # 这里把训练集和测试集分开了，也可以换成两个csv文件
data_test_mark = data_stamp[:, :]



train_loader, x_train, y_train, x_train_mark, y_train_mark = tslib_data_loader(window, length_size, batch_size, data_train, data_train_mark)
test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(window, length_size, batch_size, data_test, data_test_mark)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 20  # 训练迭代次数
learning_rate = 0.0001  # 学习率
scheduler_patience = int(0.25 * num_epochs)  # 转换为整数  学习率调整的patience
early_patience = 0.2  # 训练迭代的早停比例 即patience=0.25*num_epochs
class Config:
    def __init__(self):
        # basic
        self.seq_len = window  # input sequence length
        self.label_len = int(window / 2)  # start token length
        self.pred_len = length_size  # 预测序列长度
        self.freq = 'b'  # 时间的频率，
        # 模型训练
        self.batch_size = batch_size  # 批次大小
        self.num_epochs = num_epochs  # 训练的轮数
        self.learning_rate = learning_rate  # 学习率
        self.stop_ratio = early_patience  # 早停的比例
        # 模型 define
        self.dec_in = data_dim  # 解码器输入特征数量, 输入几个变量就是几
        self.enc_in = data_dim  # 编码器输入特征数量
        self.c_out = 1  # 输出维度##########这个很重要
        # 模型超参数
        self.d_model = 32  # 模型维度
        self.n_heads = 8  # 多头注意力头数
        self.dropout = 0.1  # 丢弃率
        self.e_layers = 3  # 编码器块的数量
        self.d_layers = 3  # 解码器块的数量
        self.d_ff = 64  # 全连接网络维度
        self.factor = 5  # 注意力因子
        self.activation = 'gelu'  # 激活函数
        self.channel_independence = 0  # 频道独立性，0:频道依赖，1:频道独立

        self.top_k = 5  # TimesBlock中的参数
        self.num_kernels = 6  # Inception中的参数
        self.distil = 1  # 是否使用蒸馏，1为True
        # 一般不需要动的参数
        self.embed = 'timeF'  # 时间特征编码方式
        self.output_attention = 0  # 是否输出注意力
        self.task_name = 'short_term_forecast'  # 模型的任务，一般不动但是必须这个参数


def cal_eval(y_real, y_pred):
    """
    输入参数:
    y_real - numpy数组，表示测试集的真实目标值。
    y_pred - numpy数组，表示预测的结果。

    输出:
    df_eval - pandas DataFrame对象
    """

    y_real, y_pred = np.array(y_real).ravel(), np.array(y_pred).ravel()

    r2 = r2_score(y_real, y_pred)
    mse = mean_squared_error(y_real, y_pred)
    rmse = mean_squared_error(y_real, y_pred)  # RMSE and MAE are various on different scales
    mae = mean_absolute_error(y_real, y_pred)
    mape = mean_absolute_percentage_error(y_real, y_pred) * 100  # Note that dataset cannot have any 0 value.

    df_eval = pd.DataFrame({'R2': r2,
                            'MSE': mse, 'RMSE': rmse,
                            'MAE': mae, 'MAPE': mape},
                           index=['Eval'])

    return df_eval

config = Config()

trained_model = Informer.Model(config).to(device)
trained_model.load_state_dict(torch.load('BIDU.pth'))

trained_model.eval()  # 模型转换为验证模式
# 预测并调整维度
pred = trained_model(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))
true = y_test[:,-length_size:,-1:].detach().cpu()
pred = pred.detach().cpu()
# 检查pred和true的维度并调整
print("Shape of true before adjustment:", true.shape)
print("Shape of pred before adjustment:", pred.shape)

# 可能需要调整pred和true的维度，使其变为二维数组
true = true[:, :, -1]
pred = pred[:, :, -1]  # 假设需要将pred调整为二维数组，去掉最后一维
# true =np.array(true)
# 假设需要将true调整为二维数组

print("Shape of pred after adjustment:", pred.shape)
print("Shape of true after adjustment:", true.shape)

 #这段代码是为了重新更新scaler，因为之前定义的scaler是是十六维，这里重新根据目标数据定义一下scaler
y_data_test_inverse = scaler.fit_transform(np.array(data_target).reshape(-1, 1))
pred_uninverse = scaler.inverse_transform(pred[:, -1:])    #如果是多步预测， 选取最后一列
true_uninverse = scaler.inverse_transform(true[:, -1:])

true, pred = true_uninverse, pred_uninverse
df_eval = cal_eval(true, pred)  # 评估指标dataframe
print(df_eval)

df_pred_true = pd.DataFrame({'Predict': pred.flatten(), 'Real': true.flatten()})
df_pred_true.plot(figsize=(12, 4))
plt.show()
# 将真实值和预测值合并为一个 DataFrame
result_df = pd.DataFrame({'真实值': true.flatten(), '预测值': pred.flatten()})
# 保存 DataFrame 到一个 CSV 文件
result_df.to_csv('真实值与预测值.csv', index=False, encoding='utf-8')
# 打印保存成功的消息
print('真实值和预测值已保存到真实值与预测值.csv文件中。')