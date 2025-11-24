# 可视化图片
import matplotlib.pyplot as plt
import numpy as np
import random
datapath = '../data/generated_data_1/cloudy_dataset_diffusion_best.h5'

# data = np.load(datapath,allow_pickle=True)
# print(data.shape)

import h5py

# filename = "./models/cloudy_dataset_diffusion_best.h5"

with h5py.File(datapath, "r") as f:
    print(f.keys())       # 查看文件里有哪些 dataset
    data = f["data"][:]   # 读取整个 dataset 到内存（numpy array）
    print(data.shape)
X = list(range(0,data.shape[1]))
data = data.reshape(data.shape[0],data.shape[1])

samples = random.sample(list(range(0,130000)),k=5)
for i in samples:
    plt.figure(figsize=(12,8))
    plt.plot(X,data[i],color='blue')
    plt.savefig(f"{i}.png",dpi=150)
    plt.close()