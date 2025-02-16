import torch
import joblib
import numpy as np
import pandas as pd
from models import Informer
from utils.timefeatures import time_features
from torch.utils.data import DataLoader, TensorDataset
from flask import Flask, request, jsonify

# 设备选择
device = torch.device("cuda")

# 定义 Config（与训练时保持一致）
class Config:
    def __init__(self):
        self.seq_len = 30
        self.label_len = 15
        self.pred_len = 1
        self.freq = "b"
        self.batch_size = 32
        self.num_epochs = 10
        self.learning_rate = 0.0001
        self.stop_ratio = 0.25
        self.dec_in = 5
        self.enc_in = 5
        self.c_out = 1
        self.d_model = 32
        self.n_heads = 8
        self.dropout = 0.1
        self.e_layers = 3
        self.d_layers = 3
        self.d_ff = 64
        self.factor = 5
        self.activation = "gelu"
        self.channel_independence = 0
        self.top_k = 5
        self.num_kernels = 6
        self.distil = 1
        self.embed = "timeF"
        self.output_attention = 0
        self.task_name = "short_term_forecast"

config = Config()

def tslib_data_loader(window, length_size, batch_size, data, data_mark):
    seq_len = window
    sequence_length = seq_len + length_size
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

# 加载模型
model = Informer.Model(config).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
scaler = joblib.load("scaler.save")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data['data'])

    print(df)

    data_target = df['Target']
    data = df[df.columns.drop('date')]

    # 数据预处理
    df_stamp = df[["date"]]
    df_stamp["date"] = pd.to_datetime(df_stamp["date"])
    data_stamp = time_features(df_stamp, timeenc=1, freq="B")
    data_inverse = scaler.fit_transform(np.array(data))

    test_loader, x_test, y_test, x_test_mark, y_test_mark = tslib_data_loader(30, 1, 32, data_inverse, data_stamp)

    pred = model(x_test.to(device), x_test_mark.to(device), y_test.to(device), y_test_mark.to(device))
    pred = pred.detach().cpu()

    pred = pred[:, :, -1]

    scaler.fit_transform(np.array(data_target).reshape(-1, 1))
    pred_uninverse = scaler.inverse_transform(pred[:, -1:])


    # 获取预测值
    predicted_value = pred_uninverse[0][0]

    # 返回结果
    return jsonify({
        "code": 200,
        "data": {
            "val": predicted_value
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)