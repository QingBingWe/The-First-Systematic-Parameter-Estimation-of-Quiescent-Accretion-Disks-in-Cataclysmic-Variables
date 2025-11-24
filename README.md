
## 训练参数
### vae 
* epochs=200
* latent_dim=256
* seq_length=3820 (是lamost数据集和sdss数据集中最小的序列长度)
### wgangp
* epochs=5000
* latent_dim=256
* seq_length=3820
### diffusion
* epochs=200
* num_steps=1000(扩散步数)
* seq_length=3820
## 生成使用的数据集
### vae和wgangp(试了128和64，256的效果是最好的)
* cloudy_dataset_256.npy (因为latent_dim=256)
* index256.txt
### diffusion
* cloudy_dataset.npy
* index.txt
## 参数量
### diffusion
![alt text](images/image-2.png)
### wgangp
![alt text](images/image-3.png)
### vae
![alt text](images/image-4.png)
# 文件结构
```
.
|____checkpoints_new
| |____vae  // 训练好的模型，以及训练过程的loss曲线图
| |____diffusion  //训练好的模型，以及训练过程的loss曲线图
| |____wgangp // 训练好的模型，以及训练过程的loss曲线图
|____param_table.csv  //送入cloudy的参数表
|____images   //README.md中的图片
|____models
| |____diffusion.py //扩散模型训练和生成的入口
| |____wgan_gp.py   //wgangp训练和生成的入口
| |____vae.py      //VAE模型训练和生成的入口
|____experiments_new   
| |____vae_sdss  //vae处理过后的cloudy数据和sdss的匹配结果
| | |____sdss_vae_cloudy.csv  
| | |____real   // 真实flux的匹配图像
| | |____normalized  //归一化后的
| |____diffusion_sdss //diffusion处理过后的cloudy数据和sdss的匹配结果
| | |____real
| | |____normalized
| | |____sdss_diffusion_cloudy.csv
| |____wgangp_lamost  //wgangp处理过后的cloudy数据和lamost的匹配结果
| | |____real
| | |____normalized
| | |____lamost_wgangp_cloudy.csv
| |____vae_lamost  //vae处理过后的cloudy数据和lamost的匹配结果
| | |____real
| | |____normalized
| | |____lamost_vae_cloudy.csv
| |____lamost_similiest //lamost和三个模型中最小的mse的匹配结果（可从中选择图片放入论文中）
| | |____lamost_finally_data.csv
| | |____real
| | |____normalized
| |____diffusion_lamost //diffusion处理过后的cloudy数据和lamost的匹配结果
| | |____real
| | |____normalized
| | |____lamost_diffusion_cloudy.csv
| |____sdss_similiest //sdss和三个模型中最小的mse的匹配结果（可从中选择图片放入论文中）
| | |____real
| | |____normalized
| | |____sdss_finally_data.csv
| |____wgangp_sdss  //wgangp处理过后的cloudy数据和sdss的匹配结果
| | |____real
| | |____normalized
| | |____sdss_wgangp_cloudy.csv
|____parameter_sdss_and_lamost.csv  //和数据集最匹配的数据制成的参数表（可用于绘图，可选择一些数据放入论文中）
|____data
| |____SDSS_DATA_PROCESS.py  //用于预处理sdss
| |____generated_data_new
| | |____wganGP_5000_cloudy_dataset_256.npy  //wgangp对cloudy风格迁移后生成的数据
| | |____vae_cloudy_dataset_256.npy //vae对cloudy风格迁移后生成的数据
| | |____cloudy_dataset_diffusion_best.h5 //diffusion对cloudy风格迁移后生成的数据
| |____Lamost_process.py //用于预处理lamost
| |____processed_2  //预处理后的sdss和lamost数据
| | |____sdss.txt
| | |____sdss_y.npy
| | |____sdss_x.npy
| | |____lamost_y.npy
| | |____lamost.txt
| | |____lamost_x.npy
| |____cloudy_data  //cloudy生成的数据
| | |____cloudy_dataset_256.npy //wgangp和vae使用
| | |____cloudy_index256.txt
| | |____cloudy_index.txt
| | |____cloudy_dataset.npy  //diffusion使用
| |____raw  //原始数据
| | |____SDSS
| | |____Lamost
| |____only_used  //提取出来的光谱
| | |____SDSS
| | |____Lamost
|____utils
| |____calculate_avg_mse.py  
| |____visilized_img.py  //用于制图，可视化
| |____Blackbody_Temperature.png  //黑体温度的图
| |____calculate_smallest_mse.py  //选择最小的smallest
| |____from_file_to_param.py  //制作参数表
| |____plt_img.py
| |____pattern_match_faster.py // 模式匹配
| |____Luminosity.png  //Luminosity的图
|____README.md
```
lamost

| model     | normalized_mse | mse          |
| --------- | -------------- | ------------ |
| diffusion | 0.003159       | 5.263012e+04 |
| vae       | 0.025629       | 6.920279e+05 |
| wgan-gp   | 0.019233       | 5.028703e+05 |

sdss

| model     | normalized_mse | mse        |
| --------- | -------------- | ---------- |
| diffusion | 0.003672       | 128.646354 |
| vae       | 0.018931       | 405.388269 |
| wgan-gp   | 0.013802       | 460.813234 |

