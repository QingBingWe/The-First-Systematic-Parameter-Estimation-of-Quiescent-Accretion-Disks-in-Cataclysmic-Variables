import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import h5py
from astropy.io import fits
import random
import scipy.interpolate as spi

def compute_mse_chunked(sdss_data, diffusion_data, chunk_size=2000):
    """
    分批计算 MSE，避免内存爆炸
    sdss_data: (N, L)
    diffusion_data: (M, L)
    chunk_size: 每次处理多少 diffusion 样本
    return:
        min_mse: (N,)
        min_index: (N,)
    """
    N, L = sdss_data.shape
    M = diffusion_data.shape[0]

    min_mse = np.full(N, np.inf)
    min_index = np.full(N, -1, dtype=int)

    for start in tqdm(range(0, M, chunk_size), desc="分块计算 MSE"):
        end = min(start + chunk_size, M)
        chunk = diffusion_data[start:end]  # (chunk_size, L)

        # (N, 1, L) - (1, chunk_size, L) -> (N, chunk_size, L)
        mse_matrix = np.mean((sdss_data[:, None, :] - chunk[None, :, :]) ** 2, axis=-1)  # (N, chunk_size)

        # 找到当前分块的最小值
        local_min = np.min(mse_matrix, axis=1)
        local_idx = np.argmin(mse_matrix, axis=1) + start

        # 更新全局最小值
        mask = local_min < min_mse
        min_mse[mask] = local_min[mask]
        min_index[mask] = local_idx[mask]

    return min_mse, min_index


def plot_one(i, sdss_name, X_data,sdss_data, gen_data, SAVE_DIR,path):
    
    data = np.loadtxt(os.path.join(path, sdss_name), comments='#')  # 自动跳过 # 开头的行
    X = data[:, 0]  # 第一列
    Y = data[:, 1]    # 第二列
    # 归一化
    Y_max = Y.max()
    Y_min = Y.min()
    X.tolist()
   

    gen_data_real = [i*(Y_max-Y_min)+Y_min for i in gen_data]
    list_sample = random.sample(list(range(10,len(Y)-10)),k=3820-20)
    list_sample = sorted(list_sample)
    plt.figure(figsize=(16, 8), dpi=80)
    plt.plot(X, Y, color='black', label='Spectrum From LAMOST',linewidth=0.5)
    plt.plot(X_data, gen_data_real, color='red', label='Generated Specitrum by CLOUDY and DIFFUSION',linestyle='dashed',linewidth=0.5)
    plt.xlabel("Wavelength(Å)")
    plt.ylabel("Relative flux")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/real/{sdss_name}.png", dpi=150)
    plt.close()
    
    plt.figure(figsize=(16, 8), dpi=80)
    plt.plot(X_data, sdss_data, color='black', label='Spectrum From LAMOST',linewidth=0.5)
    plt.plot(X_data, gen_data, color='red', label='Generated Specitrum by CLOUDY and DIFFUSION',linestyle='dashed',linewidth=0.5)
    plt.xlabel("Wavelength(Å)")
    plt.ylabel("Relative flux")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/normalized/{sdss_name}.png", dpi=150)
    plt.close()

    return (Y_max-Y_min)**2

      



if __name__ == "__main__":
    # ===================== 数据加载 =====================
    sdss_data = np.load("../data/processed_2/SDSS_y.npy")   # (407, 3820)
    diffusion_data = np.load("../data/generated_data_new/wganGP_5000_cloudy_dataset_256.npy")[:, :, 0]  # (135195, 3820)
    x_data = np.load("../data/processed_2/SDSS_x.npy")
    real_dataset_path = '../data/only_used/SDSS'
    
    # with h5py.File("../data/generated_data_new/cloudy_dataset_diffusion_best.h5", "r") as f:
    #     print(f.keys())       # 查看文件里有哪些 dataset
    #     diffusion_data = f["data"][:]

    SAVE_DIR = '../experiments_new/wgangp_SDSS'
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR,'real'),exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, 'normalized'), exist_ok=True)

    with open("../data/cloudy_data/cloudy_index.txt", "r") as f:
        cloudy_file_names = [line.strip() for line in f.readlines()]

    with open("../data/processed_2/SDSS_file_name.txt", "r") as f:
        sdss_name = [line.strip() for line in f.readlines()]

    print("SDSS 数据:", sdss_data.shape)
    print("VAE 数据:", diffusion_data.shape)
    print(len(cloudy_file_names))

    # ===================== 分批计算 MSE =====================
    min_mse, min_index = compute_mse_chunked(sdss_data, diffusion_data, chunk_size=1000)

    # ===================== 保存匹配结果 =====================
    all_data = []
    for i, name in enumerate(sdss_name):
        all_data.append({
            "sdss_file_name": name,
            "cloudy_file_name": cloudy_file_names[min_index[i]],
            "mse": min_mse[i],
            'index':min_index[i] # 从0开始
        })

    

    # ===================== 绘图 =====================
    N_plot = sdss_data.shape[0] 
    print(f"开始绘制前 {N_plot} 个样本对比图 ...")
    results = Parallel(n_jobs=8)(
        delayed(plot_one)(i, sdss_name[i], x_data[i],sdss_data[i], diffusion_data[min_index[i]], SAVE_DIR,real_dataset_path)
        for i in tqdm(range(min(N_plot, len(sdss_data))))
    )

    for i, val in enumerate(results):
        all_data[i]['real_mse'] = val*(all_data[i]['mse'])

    df = pd.DataFrame(all_data)
    df.to_csv(f"{SAVE_DIR}/SDSS_wgangp_cloudy.csv", index=False)
    print(f"CSV 已保存到 {SAVE_DIR}/SDSS_wgangp_cloudy.csv")
    print("绘图完成 ✅")
